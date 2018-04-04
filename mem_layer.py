import keras.backend as K
from keras.engine.topology import (
    Layer, _to_list, _collect_previous_mask, _is_all_none, _collect_input_shape)

import copy

from memory_with_prob import Memory


class MemLayer(Layer):

    def __init__(self, choose_k, mem_size, num_classes, **kwargs):
        self.choose_k = choose_k
        self.mem_size = mem_size
        self.num_classes = num_classes
        super(MemLayer, self).__init__(**kwargs)

    def build(self, inputs):
        assert type(inputs) is list and len(inputs)==2
        x, y = inputs
        x_shape = x.shape.as_list()[1]

        self.memory = Memory(
            key_dim=x_shape,
            memory_size=self.mem_size,
            vocab_size=2,
            choose_k=self.choose_k,
            num_classes=self.num_classes,
        )

        self.closest_label_train, self.probs_train, _, self.teacher_loss_train = (
            self.memory.query(x, y, use_recent_idx=True)
        )

        self.closest_label_pred, self.probs_pred, _, _ = (
            self.memory.query(x, None, use_recent_idx=False)
        )

    def call(self, inputs):
        x, y = inputs
        if K.learning_phase() in {1, True}:
            return [self.closest_label_train, self.probs_train, self.teacher_loss_train]
        else:
            return [self.closest_label_pred, self.probs_pred, self.teacher_loss_train]

    def compute_output_shape(self, input_shapes):
        x_shape = input_shapes[0]
        return [(None, ), (None, self.num_classes), (None, )]

    def __call__(self, inputs, **kwargs):
        if isinstance(inputs, list):
            inputs = inputs[:]

        with K.name_scope(self.name):
            # Raise exceptions in case the input is not compatible
            # with the input_spec specified in the layer constructor.
            self.assert_input_compatibility(inputs)

            # Handle laying building (weight creating, input spec locking).
            if not self.built:
                self.build(inputs)
                self.built = True

            # Handle mask propagation.
            previous_mask = _collect_previous_mask(inputs)
            user_kwargs = copy.copy(kwargs)
            if not _is_all_none(previous_mask):
                # The previous layer generated a mask.
                if has_arg(self.call, 'mask'):
                    if 'mask' not in kwargs:
                        # If mask is explicitly passed to __call__,
                        # we should override the default mask.
                        kwargs['mask'] = previous_mask
            # Handle automatic shape inference (only useful for Theano).
            input_shape = _collect_input_shape(inputs)

            # Actually call the layer, collecting output(s), mask(s), and shape(s).
            output = self.call(inputs, **kwargs)
            output_mask = self.compute_mask(inputs, previous_mask)

            # If the layer returns tensors from its inputs, unmodified,
            # we copy them to avoid loss of tensor metadata.
            output_ls = _to_list(output)
            inputs_ls = _to_list(inputs)
            output_ls_copy = []
            for x in output_ls:
                if x in inputs_ls:
                    x = K.identity(x)
                output_ls_copy.append(x)
            if len(output_ls_copy) == 1:
                output = output_ls_copy[0]
            else:
                output = output_ls_copy

            # Inferring the output shape is only relevant for Theano.
            if all([s is not None for s in _to_list(input_shape)]):
                output_shape = self.compute_output_shape(input_shape)
            else:
                if isinstance(input_shape, list):
                    output_shape = [None for _ in input_shape]
                else:
                    output_shape = None

            if not isinstance(output_mask, (list, tuple)) and len(output_ls) > 1:
                # Augment the mask to match the length of the output.
                output_mask = [output_mask] * len(output_ls)

            # Add an inbound node to the layer, so that it keeps track
            # of the call and of all new variables created during the call.
            # This also updates the layer history of the output tensor(s).
            # If the input tensor(s) had not previous Keras history,
            # this does nothing.
            self._add_inbound_node(input_tensors=inputs, output_tensors=output,
                                   input_masks=previous_mask, output_masks=output_mask,
                                   input_shapes=input_shape, output_shapes=output_shape,
                                   arguments=user_kwargs)

            # Apply activity regularizer if any:
            if hasattr(self, 'activity_regularizer') and self.activity_regularizer is not None:
                regularization_losses = [self.activity_regularizer(x) for x in _to_list(output)]
                self.add_loss(regularization_losses, _to_list(inputs))
        return output

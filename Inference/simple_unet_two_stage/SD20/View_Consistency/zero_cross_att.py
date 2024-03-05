import torch

# def rearrange_3(tensor, f):
#     F, D, C = tensor.size()
#     return torch.reshape(tensor, (F // f, f, D, C))


# def rearrange_4(tensor):
#     B, F, D, C = tensor.size()
#     return torch.reshape(tensor, (B * F, D, C))

def rearrange_3(tensor, f, g):
    F, D, C = tensor.size()
    # img1_p1, img2_p1, img1_p2, img2_p2, copy [img1_p1, img2_p1, img1_p2, img2_p2]
    # to [img1_p1, img2_p1, img3_p1, img4_p1], [img1_p2, img2_p2, img3_p2, img4_p2], xxx
    t = torch.reshape(tensor, (F // f, f, D, C))
    # to [[img1_p1, img2_p1], [img3_p1, img4_p1]]
    t = torch.reshape(t, (F // f, f // g, g, D, C))
    return t


def rearrange_4(tensor):
    B, F, G, D, C = tensor.size()
    return torch.reshape(tensor, (B * F * G, D, C))

# Current implementation only ensure the group consistency, but no consistency out the group.
# maybe use sliding window, and add the two attention value for overlapping frames.
class CrossFrameAttnProcessor:
    """
    Cross frame attention processor. Each frame attends the first frame.

    Args:
        batch_size: The number that represents actual batch size, other than the frames.
            For example, calling unet with a single prompt and num_images_per_prompt=1, batch_size should be equal to
            2, due to classifier-free guidance.
    """

    def __init__(self, video_length, group_size, sub_batch_bmm=None):
        self.video_length = video_length
        self.group_size = group_size
        self.sub_batch_bmm = sub_batch_bmm

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # [1, 2, 3][4, 5, 6]
        # [1, 2][3, 4][5, 6]
        # Cross Frame Attention
        if not is_cross_attention:
            # img1_p1, img2_p1, img1_p2, img2_p2, copy [img1_p1, img2_p1, img1_p2, img2_p2]
            # video_length = key.size()[0] // self.view_batch
            # if self.cfg_used:
            #     video_length = video_length // 2
            first_frame_index = [self.group_size // 2] * self.group_size

            # print(self.video_length, key.size()[0])

            # rearrange keys to have batch and frames in the 1st and 2nd dims respectively
            key = rearrange_3(key, self.video_length, self.group_size)
            key = key[:, :, first_frame_index]
            # rearrange values to have batch and frames in the 1st and 2nd dims respectively
            value = rearrange_3(value, self.video_length, self.group_size)
            value = value[:, :, first_frame_index]

            # rearrange back to original shape
            key = rearrange_4(key)
            value = rearrange_4(value)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # if self.sub_batch_bmm is not None:

        # print(query.size(), key.size())
        if self.sub_batch_bmm is not None:
            raise NotImplementedError
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
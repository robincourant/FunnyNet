import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class v_projection(nn.Module):
    def __init__(
        self,
        n_vision_dims: int = 2048,
        n_projection_dims: int = 512,
        attention_dim: int = 32,
    ):
        super(a_projection, self).__init__()
        self.vision_feat = ProjectionHead(n_vision_dims, n_projection_dims, "mlp")

        self.attention_dim = attention_dim
        self.SA = CA_SA(dim=attention_dim)
        self.pre = nn.Sequential(linear(n_projection_dims, 2))

    def forward(self, vision):
        # Vision projection
        vision_feat = self.vision_feat(vision)

        # CA
        b = vision_feat.shape[0]
        vision_feat = vision_feat.view(b, self.n, -1)
        feat = self.SA(vision_feat, vision_feat) + vision_feat
        feat = feat.view(b, -1)
        prob = self.pre(feat)

        return prob


class a_projection(nn.Module):
    def __init__(
        self,
        n_audio_dims: int = 2048,
        n_projection_dims: int = 512,
        attention_dim: int = 32,
    ):
        super(a_projection, self).__init__()
        self.audio_feat = ProjectionHead(n_audio_dims, n_projection_dims, "mlp")

        self.attention_dim = attention_dim
        self.SA = CA_SA(dim=attention_dim)
        self.pre = nn.Sequential(linear(n_projection_dims, 2))

    def forward(self, audio):
        # Audio projection
        audio_feat = self.audio_feat(audio)

        # CA
        b = audio_feat.shape[0]
        audio_feat = audio_feat.view(b, self.n, -1)
        feat = self.SA(audio_feat, audio_feat) + audio_feat
        feat = feat.view(b, -1)
        prob = self.pre(feat)

        return prob


class t_projection(nn.Module):
    def __init__(
        self,
        n_text_dims: int = 2048,
        n_projection_dims: int = 512,
        attention_dim: int = 32,
    ):
        super(t_projection, self).__init__()
        self.text_feat = ProjectionHead(n_text_dims, n_projection_dims, "mlp")

        self.attention_dim = attention_dim
        self.SA = CA_SA(dim=attention_dim)
        self.pre = nn.Sequential(linear(n_projection_dims, 2))

    def forward(self, text):
        # Text projection
        text_feat = self.text_feat(text)

        # CA
        b = text_feat.shape[0]
        text_feat = text_feat.view(b, self.n, -1)
        feat = self.SA(text_feat, text_feat) + text_feat
        feat = feat.view(b, -1)
        prob = self.pre(feat)

        return prob


class av_projection(nn.Module):
    def __init__(
        self,
        n_audio_dims: int = 2048,
        n_vision_dims: int = 1536,
        n_projection_dims: int = 512,
        attention_dim: int = 32,
    ):
        super(av_projection, self).__init__()

        self.audio_feat = ProjectionHead(n_audio_dims, n_projection_dims, "mlp")
        self.vision_feat = ProjectionHead(n_vision_dims, n_projection_dims, "mlp")

        self.attention_dim = attention_dim
        self.ZA = CA_SA(dim=attention_dim)
        self.ZV = CA_SA(dim=attention_dim)
        self.SA = CA_SA(dim=attention_dim)
        self.pre = nn.Sequential(linear(n_projection_dims, 2))

    def forward(self, audio, vision):
        # Audio projection
        audio_feat = self.audio_feat(audio)

        # Vision projection
        x = torch.mean(vision[:, 1:], dim=1)
        vision = torch.cat((x, vision[:, 0]), dim=1)
        vis_feat = self.vision_feat(vision)

        # CA
        b = audio_feat.shape[0]
        z_feat = torch.cat(
            (F.normalize(audio_feat, dim=1), F.normalize(vis_feat, dim=1)), dim=1
        )
        z_feat = z_feat.view(b, -1, self.attention_dim)
        feat_ZA = self.ZA(z_feat, audio_feat.view(b, -1, self.attention_dim))
        feat_ZV = self.ZV(z_feat, vis_feat.view(b, -1, self.attention_dim))

        # SA
        feat = feat_ZA + feat_ZV
        feat = self.SA(feat, feat) + feat
        feat1, feat2 = feat.chunk(2, dim=1)
        feat = feat1.view(b, -1) + feat2.view(b, -1)
        prob = self.pre(feat)

        return audio_feat, vis_feat, prob


class avt_projection(nn.Module):
    def __init__(
        self,
        n_audio_dims: int = 2048,
        n_vision_dims: int = 1536,
        n_text_dims: int = 1536,
        n_projection_dims: int = 512,
        attention_dim: int = 32,
    ):
        super(avt_projection, self).__init__()

        self.audio_feat = ProjectionHead(n_audio_dims, n_projection_dims, "mlp")
        self.vision_feat = ProjectionHead(n_vision_dims, n_projection_dims, "mlp")
        self.text_feat = ProjectionHead(n_text_dims, n_projection_dims, "mlp")

        self.attention_dim = attention_dim
        self.ZA = CA_SA(dim=attention_dim)
        self.ZV = CA_SA(dim=attention_dim)
        self.ZT = CA_SA(dim=attention_dim)
        self.SA = CA_SA(dim=attention_dim)
        self.pre = nn.Sequential(linear(n_projection_dims, 2))

    def forward(self, audio, vision, text):
        # Audio projection
        audio_feat = self.audio_feat(audio)

        # Vision projection
        x = torch.mean(vision[:, 1:], dim=1)
        vision_x = torch.cat((x, vision[:, 0]), dim=1)
        vis_feat = self.vision_feat(vision_x)

        # Text projection
        x = torch.mean(text[:, 1:], dim=1)
        text_x = torch.cat((x, text[:, 0]), dim=1)
        text_feat = self.text_feat(text_x)

        # CA
        b = audio_feat.shape[0]
        z_feat = torch.cat(
            (
                F.normalize(audio_feat, dim=1),
                F.normalize(vis_feat, dim=1),
                F.normalize(text_feat, dim=1),
            ),
            dim=1,
        )
        z_feat = z_feat.view(b, -1, self.attention_dim)
        feat_ZA = self.ZA(z_feat, audio_feat.view(b, -1, self.attention_dim))
        feat_ZV = self.ZV(z_feat, vis_feat.view(b, -1, self.attention_dim))
        feat_ZT = self.ZT(z_feat, text_feat.view(b, -1, self.attention_dim))

        # SA
        feat = feat_ZA + feat_ZV + feat_ZT
        feat = self.SA(feat, feat) + feat
        feat1, feat2, feat3 = feat.chunk(3, dim=1)
        feat = feat1.view(b, -1) + feat2.view(b, -1) + feat3.view(b, -1)
        prob = self.pre(feat)

        return audio_feat, vis_feat, text_feat, prob


class avf_projection(nn.Module):
    def __init__(
        self,
        n_audio_dims: int = 2048,
        n_vision_dims: int = 1536,
        n_face_dims: int = 512,
        n_projection_dims: int = 512,
        attention_dim: int = 32,
    ):
        super(avf_projection, self).__init__()
        self.n = 16
        self.audio_feat = ProjectionHead(n_audio_dims, n_projection_dims, "mlp")
        self.vision_feat = ProjectionHead(n_vision_dims, n_projection_dims, "mlp")
        self.face_feat1 = nn.LSTM(n_face_dims, 64, batch_first=True)
        self.face_feat2 = ProjectionHead(n_face_dims, n_projection_dims, "mlp")

        self.ZA = CA_SA(dim=attention_dim)
        self.ZV = CA_SA(dim=attention_dim)
        self.ZF = CA_SA(dim=attention_dim)
        self.SA = CA_SA(dim=attention_dim)
        self.pre = nn.Sequential(linear(n_projection_dims, 2))

    def forward(self, audio, vision, face):
        audio_feat = self.audio_feat(audio)
        x = torch.mean(vision[:, 1:], dim=1)
        vision = torch.cat((x, vision[:, 0]), dim=1)
        vis_feat = self.vision_feat(vision)
        face_feat, h = self.face_feat1(face)
        face_feat = face_feat.reshape(face.shape[0], -1)
        face_feat = self.face_feat2(face_feat)
        # CA
        b = audio_feat.shape[0]
        z_feat = torch.cat(
            (
                F.normalize(audio_feat, dim=1),
                F.normalize(vis_feat, dim=1),
                F.normalize(face_feat, dim=1),
            ),
            dim=1,
        )
        z_feat = z_feat.view(b, 3 * self.n, -1)
        feat_ZA = self.ZA(z_feat, audio_feat.view(b, self.n, -1))
        feat_ZV = self.ZV(z_feat, vis_feat.view(b, self.n, -1))
        feat_ZF = self.ZF(z_feat, face_feat.view(b, self.n, -1))
        # SA
        feat = feat_ZA + feat_ZV + feat_ZF
        feat = self.SA(feat, feat) + feat
        feat1, feat2, feat3 = feat.chunk(3, dim=1)
        feat = feat1.view(b, -1) + feat2.view(b, -1) + feat3.view(b, -1)
        prob = self.pre(feat)

        return audio_feat, vis_feat, face_feat, prob


#########################################################################################
#########################################################################################


class linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
        )
        self.layer2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x) * self.layer2(x)

        return x


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, head_type: str):
        super().__init__()
        if head_type == "mlp":
            self.projection = LinearHead(embedding_dim, projection_dim)
        elif head_type == "mlp":
            self.projection = MLPHead(embedding_dim, projection_dim, 0.1)

    def forward(self, x):
        x = self.projection(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: int):
        """
        https://github.com/moein-shariatnia/OpenAI-CLIP
        """
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class LinearHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int):
        """
        https://github1s.com/facebookresearch/ImageBind
        """
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-6),
            # SelectElement(index=0),
            nn.Linear(embedding_dim, projection_dim, bias=False),
        )
        self.postprocessor = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=1 / 0.07, learnable=False),
        )

    def forward(self, x):
        x = self.head(x)
        x = self.postprocessor(x)
        return x


class LearnableLogitScaling(nn.Module):
    def __init__(
        self,
        logit_scale_init: float = 1 / 0.07,
        learnable: bool = True,
        max_logit_scale: float = 100,
    ) -> None:
        super().__init__()
        self.max_logit_scale = max_logit_scale
        self.logit_scale_init = logit_scale_init
        self.learnable = learnable
        log_logit_scale = torch.ones([]) * np.log(self.logit_scale_init)
        if learnable:
            self.log_logit_scale = nn.Parameter(log_logit_scale)
        else:
            self.register_buffer("log_logit_scale", log_logit_scale)

    def forward(self, x):
        return torch.clip(self.log_logit_scale.exp(), max=self.max_logit_scale) * x

    def extra_repr(self):
        st = (
            f"logit_scale_init={self.logit_scale_init},learnable={self.learnable}, "
            f"max_logit_scale={self.max_logit_scale}"
        )
        return st


class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)


class SelectElement(nn.Module):
    def __init__(self, index) -> None:
        super().__init__()
        self.index = index

    def forward(self, x):
        assert x.ndim >= 3
        return x[:, self.index, ...]


#########################################################################################
#########################################################################################
class CA_SA(nn.Module):
    def __init__(self, dim=32):
        super(CA_SA, self).__init__()
        self.dim = dim
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.Q = nn.Linear(dim, dim)
        self.attend = nn.Softmax(dim=-1)

    def forward(self, feat1, feat2):
        Q = self.Q(feat1)
        K = self.K(feat2)
        V = self.V(feat2)
        dots = torch.bmm(Q, K.permute(0, 2, 1))
        attn = self.attend(dots)
        out = torch.bmm(attn, V)
        return out


class CA_SA_vis(nn.Module):
    def __init__(self, dim=32):
        super(CA_SA_vis, self).__init__()
        self.dim = dim
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.Q = nn.Linear(dim, dim)
        self.attend = nn.Softmax(dim=-1)

    def forward(self, feat1, feat2):
        K = self.K(feat2)
        V = self.V(feat2)
        Q = self.Q(feat1)
        attn = torch.bmm(Q, K.permute(0, 2, 1))
        attn = self.attend(attn)
        out = torch.bmm(attn, V)

        return out, attn


class CA_SA_vis_v2(nn.Module):
    def __init__(self, dim=32):
        super(CA_SA_vis_v2, self).__init__()
        self.dim = dim
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.Q = nn.Linear(dim, dim)
        self.attend = nn.Softmax(dim=-1)

    def forward(self, feat1, feat2):
        K = self.K(feat2)
        Q = self.Q(feat1)
        Q = F.normalize(Q, dim=2)
        K = F.normalize(K, dim=2)
        dots = torch.bmm(Q, K.permute(0, 2, 1))
        attn = self.attend(dots)

        return attn

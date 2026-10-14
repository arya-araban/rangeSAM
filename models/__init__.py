from .rangesam import RangeSAM


def build_model(config, pretrained_path=None):
    """Build RangeSAM from a config dict (one of the files in configs/)."""
    return RangeSAM(
        in_channels=config['in_channels'],
        n_cls=config['n_classes'],
        backbone=config['vit_backbone'],
        image_size=config['image_size'],
        pretrained_path=pretrained_path,
        new_patch_size=config['patch_size'],
        new_patch_stride=config['patch_stride'],
        reuse_pos_emb=config.get('reuse_pos_emb', False) and pretrained_path is not None,
        conv_stem=config.get('conv_stem', 'ConvStem'),
        stem_base_channels=config.get('stem_base_channels', 32),
        stem_hidden_dim=config['D_h'],
        skip_filters=config['skip_filters'],
        decoder='up_conv',
        up_conv_d_decoder=config['D_h'],
        up_conv_scale_factor=config['patch_stride'],
        prompt_embed_dim=config['prompt_embed_dim'],
        transformer_depth=config['transformer_depth'],
        num_heads=config['num_heads'],
        mlp_dim=config['mlp_dim'],
        label_enforcement=config.get('label_enforcement', True),
        learnable_prompt_scale=config.get('learnable_prompt_scale', True),
    )

from .coswin import CoSwinTransformer

def build_model(img_size, n_classes, args):
    patch_size = 2 if img_size == 32 else 4   

    if args.model =="coswin":
        return CoSwinTransformer(img_size=img_size, 
                                window_size=4, 
                                drop_path_rate=0.1, 
                                patch_size=patch_size, 
                                mlp_ratio=2, depths=[2, 6, 4], 
                                num_heads= [3, 6, 12], 
                                num_classes=n_classes, 
                                )
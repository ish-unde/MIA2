import numpy as np 

def normalize_intensity(img, percentile_clip=(1, 99)):
    """
    Normalize using only foreground voxels (non-zero / brain region),
    with optional percentile clipping to suppress outliers.
    """
    arr = img.numpy()

    # Use only foreground voxels for statistics
    foreground = arr[arr > 0]

    # Clip outliers before normalizing (scanner artifacts, bright spots)
    low, high = np.percentile(foreground, percentile_clip)
    arr_clipped = np.clip(arr, low, high)

    # Z-score using foreground statistics only
    foreground_clipped = arr_clipped[arr_clipped > 0]
    mean = foreground_clipped.mean()
    std  = foreground_clipped.std()
    arr_norm = (arr_clipped - mean) / std

    # Keep background at 0
    arr_norm[arr == 0] = 0

    return img.new_image_like(arr_norm.astype(np.float32))
    
def preprocess_mri(volume_path, apply_gaussian=False, sigma=0.5,
                   register_to=None, mask_paths=None, output_dir=None, verbose=True):
    """
    Args:
        volume_path:  path to brain .nii.gz
        mask_paths:   dict of name -> path, e.g. {'catheter': 'sub001_catheter.nii.gz'}
        register_to:  path to fixed/template image
        output_dir:   where to save outputs
    """
    import ants
    import numpy as np
    from scipy.ndimage import gaussian_filter

    # ── Load everything as ANTs images (preserves full spatial metadata) ──────
    moving = ants.image_read(volume_path)

    masks = {}
    if mask_paths:
        for name, path in mask_paths.items():
            masks[name] = ants.image_read(path)

    # ── N4 Bias Field Correction (native ANTs — no format conversion)
    if verbose:
        print("  Applying N4 bias-field correction...")
    moving_n4 = ants.n4_bias_field_correction(moving)

    if apply_gaussian:
        if verbose:
            print(f"  Applying Gaussian smoothing (sigma={sigma})...")
        arr = moving_n4.numpy()
        arr_smooth = gaussian_filter(arr, sigma=sigma)
        moving_n4 = moving_n4.new_image_like(arr_smooth)  # preserves metadata

    # intensity normalization 
    if verbose:
        print("  Normalizing intensity...")

    
    moving_norm = normalize_intensity(moving_n4, percentile_clip=(1, 99))


    # Registration 
    transform = None
    registered_masks = {}

    if register_to is not None:
        if verbose:
            print("  Performing rigid registration to template...")

        fixed = ants.image_read(register_to)

        # Rescale to [0, 1000] for registration stability
        def rescale(img):
            arr = img.numpy()
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 1000
            return img.new_image_like(arr.astype(np.float32))

        reg = ants.registration(
            fixed=rescale(fixed),
            moving=rescale(moving_norm),
            type_of_transform='Rigid',
            verbose=verbose
        )
        transform = reg['fwdtransforms']

        # Apply to volume
        moving_norm = ants.apply_transforms(
            fixed=fixed,
            moving=moving_norm,
            transformlist=transform
        )

        # Apply SAME transform to every mask
        for name, mask_img in masks.items():
            registered_masks[name] = ants.apply_transforms(
                fixed=fixed,
                moving=mask_img,
                transformlist=transform,
                interpolator='nearestNeighbor'
            )
            if verbose:
                unique = np.unique(registered_masks[name].numpy())
                print(f"  Mask '{name}' registered, unique values: {unique}")
    else:
        registered_masks = masks  # no registration, return as-is

    # saving 
    results = {'volume': moving_norm, 'masks': registered_masks, 'transform': transform}

    if output_dir:
        import os, shutil
        os.makedirs(output_dir, exist_ok=True)
        subject_id = os.path.basename(volume_path).replace('.nii.gz', '').replace('.nii', '')

        vol_out = os.path.join(output_dir, f"{subject_id}_preprocessed.nii.gz")
        ants.image_write(moving_norm, vol_out)
        if verbose:
            print(f"  Saved volume → {vol_out}")

        for name, mask_img in registered_masks.items():
            mask_out = os.path.join(output_dir, f"{subject_id}_{name}.nii.gz")
            ants.image_write(mask_img, mask_out)
            if verbose:
                print(f"  Saved mask '{name}' → {mask_out}")

        if transform:
            for t_idx, t_path in enumerate(transform):
                ext = os.path.splitext(t_path)[-1]
                dest = os.path.join(output_dir, f"{subject_id}_transform_{t_idx}{ext}")
                shutil.copy(t_path, dest)

    return results


def main(
    root_folder,
    output_folder,
    template_path=None,
    mask_names=None,
    brain_suffix="_brain.nii.gz",
    mask_suffix_template="_{name}.nii.gz",
    apply_gaussian=False,
    sigma=0.5,
    verbose=True
):
    import os, glob

    subject_dirs = sorted([
        d for d in glob.glob(os.path.join(root_folder, "*"))
        if os.path.isdir(d)
    ])

    if not subject_dirs:
        print(f"No subject folders found in {root_folder}")
        return

    print(f"Found {len(subject_dirs)} subject(s).\n")

    for idx, subject_dir in enumerate(subject_dirs):
        subject_id = os.path.basename(subject_dir)
        print(f"[{idx + 1}/{len(subject_dirs)}] {subject_id}")

        brain_path = os.path.join(subject_dir, f"{subject_id}{brain_suffix}")
        if not os.path.exists(brain_path):
            print(f"  Brain image not found, skipping.\n")
            continue

        # Build mask path dict
        mask_paths = {}
        if mask_names:
            for name in mask_names:
                p = os.path.join(subject_dir, f"{subject_id}{mask_suffix_template.format(name=name)}")
                if os.path.exists(p):
                    mask_paths[name] = p
                elif verbose:
                    print(f"  Mask '{name}' not found, skipping.")

        subject_out_dir = os.path.join(output_folder, subject_id)

        try:
            preprocess_mri(
                volume_path=brain_path,
                apply_gaussian=apply_gaussian,
                sigma=sigma,
                register_to=template_path,
                mask_paths=mask_paths if mask_paths else None,
                output_dir=subject_out_dir,
                verbose=verbose
            )
        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue

        print()

    print("Done.")


if __name__ == "__main__":
    main(
        root_folder="C:/Users/ishit/MIA/project_work/project2/MIA_2026_Project2_Data/Train",
        output_folder="C:/Users/ishit/MIA/project_work/project2/proj2_preprocessed",
        template_path="C:/Users/ishit/MIA/project_work/project2/icbm_avg_152_t1_tal_lin.nii",  # None to skip registration
        mask_names=["catheter", "artifact"],
        brain_suffix="_image.nii.gz",
        mask_suffix_template="_{name}.nii.gz",
        apply_gaussian=False,
        sigma=0.5,
        verbose=True
    )
    
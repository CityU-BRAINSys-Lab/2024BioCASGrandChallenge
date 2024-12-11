import augmentators

def LoadAugmentators(proj):
    augmentator_list = []
    for aug_name in proj.augmentators:
        aug = getattr(augmentators, aug_name)
        augmentator_list.append(aug(proj))

    return augmentator_list
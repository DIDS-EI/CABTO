is_patched = False

def patch_micro_particle_system():
    try:
        from omnigibson.systems.micro_particle_system import MicroPhysicalParticleSystem

        if hasattr(MicroPhysicalParticleSystem, '_original_sync_particle_instancers'):
            return

        MicroPhysicalParticleSystem._original_sync_particle_instancers = MicroPhysicalParticleSystem._sync_particle_instancers

        def new_sync_particle_instancers(self, idns, particle_groups, particle_counts):
            idns = [int(idn) for idn in idns]
            particle_groups = [int(group) for group in particle_groups]
            particle_counts = [int(count) for count in particle_counts]
            return self._original_sync_particle_instancers(idns, particle_groups, particle_counts)

        MicroPhysicalParticleSystem._sync_particle_instancers = new_sync_particle_instancers
    except ImportError:
        pass  # omnigibson not available, skip patching



def patch_vision_utils():
    try:
        from omnigibson.utils.vision_utils import Remapper
        
        if hasattr(Remapper, '_original_remap'):
            return
            
        Remapper._original_remap = Remapper.remap
        
        def new_remap(self, old_mapping, new_mapping, image, image_keys=None):
            if image.numel() == 0:
                return image, new_mapping
            return self._original_remap(old_mapping, new_mapping, image, image_keys)
        
        Remapper.remap = new_remap
    except ImportError:
        pass  # omnigibson not available, skip patching


def patch_cloth_prim():
    try:
        import torch as th
    except ImportError as e:
        # Skip patching if torch cannot be imported (e.g., missing libnvshmem_host.so.3)
        return
    
    try:
        from omnigibson.prims.cloth_prim import ClothPrim

        if hasattr(ClothPrim, '_original_set_particle_positions'):
            return

        ClothPrim._original_set_particle_positions = ClothPrim.set_particle_positions

        def new_set_particle_positions(self, positions, idxs=None):
            """
            Sets individual particle positions for this cloth prim

            Args:
                positions (n-array): (N, 3) numpy array, where each of the N particles' positions are expressed in (x,y,z)
                    cartesian coordinates relative to the world frame
                idxs (n-array or None): If set, will only set the requested indexed particle state
            """
            n_expected = self._n_particles if idxs is None else len(idxs)
            # assert (
            #     len(positions) <= n_expected
            # ), f"Got mismatch in particle setting size: {len(positions)}, vs. number of expected particles {n_expected}!"

            if len(positions) < n_expected:
                positions = th.cat([positions, positions[-1].repeat(n_expected - len(positions), 1)], dim=0)
            elif len(positions) > n_expected:
                positions = positions[:n_expected]

            return self._original_set_particle_positions(positions, idxs)

        ClothPrim.set_particle_positions = new_set_particle_positions
    except ImportError:
        pass  # omnigibson not available, skip patching

def apply_all_patches():
    global is_patched
    if is_patched:
        return
    is_patched = True
    patch_micro_particle_system()
    patch_vision_utils()
    patch_cloth_prim()
    try:
        import omnigibson
        print("omnigibson is patched by btgym")
    except ImportError:
        pass  # omnigibson not available, skipping patch message
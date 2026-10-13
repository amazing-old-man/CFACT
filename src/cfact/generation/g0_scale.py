"""Outcome-blind G0 amplitude from a fixed Source and current physical geometry."""
import numpy as np

def source_geometry_scale(source_poses, *, geometry_allowance_m):
    poses=np.asarray(source_poses,dtype=float)
    if poses.ndim!=3 or poses.shape[1:]!=(4,4) or len(poses)<2 or not np.isfinite(poses).all():
        raise ValueError('invalid_source_scale_geometry')
    if not np.isfinite(geometry_allowance_m) or geometry_allowance_m<0:
        raise ValueError('invalid_current_geometry_allowance')
    points=poses[:,:3,3]
    excursion=float(np.linalg.norm(points-points[0],axis=1).max())
    p90=float(np.quantile(np.linalg.norm(np.diff(points,axis=0),axis=1),.9))
    scale=min(.03,.15*excursion,4*p90,float(geometry_allowance_m))
    return {'schema':'cfact.g0_source_geometry_scale.v1','legacy_amplitude_m':.003,
            'requested_amplitude_m':float(scale),'source_excursion_m':excursion,
            'source_p90_step_m':p90,'current_geometry_allowance_m':float(geometry_allowance_m),
            'absolute_ceiling_m':.03,'outcome_blind':True,
            'actual_achieved_displacement':'measured_after_env_step_not_predicted'}

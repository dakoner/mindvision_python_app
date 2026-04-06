import cv2
import numpy as np

def optimize_placement(frame1, frame2, dx_est, dy_est):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    H, W = gray1.shape
    
    shift_x = int(round(dx_est))
    shift_y = int(round(dy_est))
    
    # Overlap region in frame1
    x_min = max(0, shift_x)
    x_max = min(W, W + shift_x)
    y_min = max(0, shift_y)
    y_max = min(H, H + shift_y)
    
    if x_max - x_min < 50 or y_max - y_min < 50:
        return dx_est, dy_est
        
    # template from center of overlap in frame1
    tw, th = int((x_max - x_min)*0.5), int((y_max - y_min)*0.5)
    cx, cy = (x_min + x_max)//2, (y_min + y_max)//2
    
    tx0 = cx - tw//2
    ty0 = cy - th//2
    tx1 = tx0 + tw
    ty1 = ty0 + th
    
    template = gray1[ty0:ty1, tx0:tx1]
    
    search_cx = cx - shift_x
    search_cy = cy - shift_y
    
    sw, sh = tw + 60, th + 60
    sx0 = max(0, search_cx - sw//2)
    sy0 = max(0, search_cy - sh//2)
    sx1 = min(W, search_cx + sw//2)
    sy1 = min(H, search_cy + sh//2)
    
    search_region = gray2[sy0:sy1, sx0:sx1]
    if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
        return dx_est, dy_est
        
    res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    if max_val < 0.3:
        return dx_est, dy_est
        
    match_x2 = sx0 + max_loc[0]
    match_y2 = sy0 + max_loc[1]
    
    opt_dx = tx0 - match_x2
    opt_dy = ty0 - match_y2
    
    return float(opt_dx), float(opt_dy)

# Create dummy frames to test
frame1 = np.zeros((100, 200, 3), dtype=np.uint8)
frame1[40:60, 140:160] = 255 # object at x=150

frame2 = np.zeros((100, 200, 3), dtype=np.uint8)
# Camera moved right by 105 pixels. Object should be at 150 - 105 = 45
frame2[40:60, 35:55] = 255 

dx, dy = optimize_placement(frame1, frame2, 100, 0)
print(f"Estimated dx: 100, Optimized dx: {dx}")

import scipy
from scipy import optimize
import itertools
from copy import deepcopy
import cv2
import numpy as np
import scipy 
from skimage.morphology import skeletonize
from skimage.graph import pixel_graph
import matplotlib.pyplot as plt
import open3d as o3d

def evaluate(edge, center, radius):
    pred = []
    for i in np.arange(0, 2*np.pi, 0.1):
        x = center[0] + radius * np.cos(i)
        y = center[1] + radius * np.sin(i)
        pred.append([x,y])

    pred = np.array(pred)

    pred_tree = scipy.spatial.cKDTree(pred)
    
    d = pred_tree.query(edge, k=1, distance_upper_bound = 3)
    
    matched = len([i for i in d[0] if i != np.inf])
    
    percent = matched/len(edge)
    
    return percent

def scale(id, size = (96,96)):
    img = cv2.imread(id)

    if img.shape[1] > img.shape[0]:
        size = (96, img.shape[1]/img.shape[0]*96)

    if img.shape[0] > img.shape[1]:
        size = (img.shape[0]/img.shape[1]*96, 96)
    
    img = cv2.resize(img, size)

    return img

def curvature(contours, thres = 0.95, k =7):

    curve = []
    edges = []
    num_edge = -1
    new_edge = True

    for contour in contours:
        
        max_len = len(contour)
        step = k
        if k >= max_len:
           step = max_len -1

        if np.linalg.norm(contour[0] - contour[-1]) <  1.5:
            is_loop = True
        else:
            is_loop = False
        
        for i, seed in enumerate(contour):
            ind1 = i + step
            ind2 = i - step
            
            if ind1 >= max_len:
                if is_loop:
                    ind1 = ind1 - max_len
                else:
                    ind1 = -1

            node1 = contour[ind1]
            node2 = contour[ind2]

            chord1 = np.linalg.norm(seed - node1)
            chord2 = np.linalg.norm(seed - node2)
            chord3 = np.linalg.norm(node1 - node2)

            if chord3/(chord1 + chord2) < thres:
                curve.append(seed)
                new_edge = True

            elif new_edge:
                edges.append(seed)
                num_edge += 1
                new_edge = False

            else:
                edges[num_edge] = np.append(edges[num_edge], seed, axis = 0)

            if (i == len(contour) - 1) and len(edges) > 1:
                if chord3/(chord1 + chord2) > thres and np.linalg.norm(contour[0] - contour[-1]) <  1.5:
                    edges[0] = np.append(edges[-1], edges[0], axis = 0)
                    edges.pop(-1)
    
    return curve, edges

def contours_filter(contours):
    
    for j, contour in enumerate(contours):
        for i in range(len(contour)-2):  
            if all(contour[i][0] == contour[i + 2][0]):
                
                new_contour = contour[::-1]
                for k, node2 in enumerate(new_contour):
                    if all(node2[0] == new_contour[k-2][0]):
                        contours[j] = contour[i+1:len(contour)-k]
                        break
                break
    
    return contours
                
def lestsq(contour):

    global x
    global y
    x = contour[:,0] 
    y = contour[:,1]

    x_m = np.mean(x)
    y_m = np.mean(y)
    
    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2       = calc_R(*center_2)
    R_2        = Ri_2.mean()
    residu_2   = np.sum((Ri_2 - R_2)**2)

    return (int(xc_2), int(yc_2)), int(R_2)

def calc_R(xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f_2(c):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(*c)
    return Ri - Ri.mean()

def ccw(A,B,C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def find_intersections(edge):

    v = edge[-1] - edge[0]
    v_x = v[1]
    v_y = v[0]
    c = edge[0]
    c_x = c[1]
    c_y = c[0]
    
    inter = []

    xs = [0,96]
    for x in xs:
        #y = slope*x + C
        y = v_y/v_x*(x - c_x) + c_y
        if 0<= y < 96:
            inter.append(np.array([y,x]))

    ys = [0,96]
    for y in ys:
        #x = (y - C)/slope
        x = v_x/v_y*(y - c_y) + c_x
        if 0 <= x < 96:
            inter.append(np.array([y,x]))

    return inter

def overlaps(img, edges):

    h,w = img.shape[0:2]
    
    dir = []
    mids = []
    masks = []

    corners = np.array([[0,0],[0,w],[h,w],[h,0]])
    
    for edge in edges:
        min_len = 10000
    
        inter = find_intersections(edge)
        
        pts = np.concatenate(([inter[0]],[inter[1]]))
        mid = edge[int(len(edge)/2)]
        mask = np.zeros_like(img)
        for corner in corners:
            if ccw(inter[0],inter[1],mid) != ccw(inter[0],inter[1],corner):
                pts = np.concatenate((pts,np.array([corner])))
                
        for j, iter in enumerate(itertools.permutations(pts)):
            dist=0
            for i in range(len(iter)-1):
                dist += np.linalg.norm(iter[i]-iter[i+1])
            
            if dist < min_len:
                choosen = iter
                min_len = dist
            
            if j == 5:
                break
            
        _=cv2.drawContours(mask, np.int32([np.array(choosen)]),0, 255, -1)

        masks.append(mask)

    return masks

def is_intersection(img, edges):
    masks = overlaps(img, edges)

    for i,mask in enumerate(masks):
        edge = edges[i-1]
        pts = [edge[1],edge[-2],edge[int(len(edge)/2)]]
        
        for pt in pts:
            
            if mask[pt[1],pt[0]] == 0:
                return False

    return True

def circle_filter(img, edges):

    img = img/255
    max_IoU = 0
    
    combinations = list(range(len(edges)))
    total_edges = np.array([point for edge in edges for point in edge])
    
    for i in itertools.permutations(range(len(edges)), r = 2):
        if i[-1] >= i[0]:
            combinations.append(i)

    for i in combinations:
        
        if type(i) == int:
            edge = edges[i]
            center, radius = lestsq(edge)
            
            IoU = evaluate(total_edges, center, radius)*0.7 + iou2(img, center, radius) *0.3
            #print('single', IoU)
            if IoU > max_IoU:
                max_IoU = IoU
                chosen = edge
        else:
            asd = is_intersection(img, [edges[i[0]], edges[i[1]]])
            
            if asd:
                edge = np.concatenate((edges[i[0]], edges[i[1]]))
                center, radius = lestsq(edge)
                
                IoU = evaluate(total_edges, center, radius)*0.5 + iou2(img, center, radius)*0.5
                #print('double', IoU)
                if IoU > max_IoU:
                    max_IoU = IoU
                    chosen = edge
        
    return chosen

def iou2(img, center,radius):
    img = img.astype(np.float32)
    img_BGR = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    total = np.sum(img/255)
    h,w = img.shape[0:2]
    epsilon = 0.001

    mask = np.zeros_like(img_BGR)
    #if center[0] < h and center[0] > 0 and center[1] < w and center[1] > 0:
    mask = cv2.circle(mask, center, radius, (255,255,255), -1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)/255
    inlier = np.sum(img/255*mask)
    outlier = np.pi*radius**2 - inlier

    percent = outlier/(inlier + epsilon)
    
    percent = inlier/total
        
    return percent

def localize_apple(mask_id):

    gray = cv2.imread(mask_id,0)
    gray = cv2.resize(gray, (96,96))
    gray = cv2.medianBlur(gray,3)
    
    edges = cv2.Canny(gray,100,200, L2gradient = True)
    edges = np.array(skeletonize(edges/255)*255).astype('uint8')
    

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if len(contour) > 5]

    contours = contours_filter(contours)
    curves, edges2 = curvature(contours, thres = 0.97, k =7)
    edges2 = [edge for edge in edges2 if len(edge) > 10]
    
    edges2 = circle_filter(gray, edges2)

    center, radius = lestsq(edges2)

    return center, radius

def orientation_computation(center, radius, poses, threshold):
    
    index = np.where(poses['score'] > threshold)[0]
    
    if len(index)  == 1:
        pose = poses['keypoints'][index[0]]
        if index == 0:
            _type = 'calyx'
        else:
            _type = 'stem'
    else:
        i = np.argmax(poses['score'])
        pose = poses['keypoints'][i]
        if i == 0:
            _type = 'calyx'
        else:
            _type = 'stem'

    y_c = center[0]
    x_c = center[1]
    r = radius
    y = y_c - pose[1]
    x = pose[0] - x_c
    r_c = np.sqrt(x**2 + y**2)
    
    if r_c > r:
        x = x/r_c*r
        y = y/r_c*r
    
    z = np.sqrt(np.abs(r**2 - x**2 - y**2))
    vec = np.array([x,y,z])

    return vec, _type

def visualizer_2d(img,  center, radius, save_id):
    img = cv2.circle(img,center,radius,(0,255,0),1)
    cv2.imshow('result',img)
    
    pressedKey = cv2.waitKey(0) & 0xFF
    if pressedKey == ord('x'):
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(save_id, img)
        cv2.destroyAllWindows()
    

def draw_geometries(pcds):
    """
    Draw Geometries
    Args:
        - pcds (): [pcd1,pcd2,...]
    """
    o3d.visualization.draw_geometries(pcds)

def get_o3d_FOR(origin=[0, 0, 0],size=10):
    """ 
    Create a FOR that can be added to the open3d point cloud
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=size)
    mesh_frame.translate(origin)
    return(mesh_frame)

def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec (): 
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)


def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the 
    z axis vector of the original FOR. The first rotation that is 
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis. 

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec (): 
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                   [np.sin(gamma),np.cos(gamma),0],
                   [0,0,1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T@vec.reshape(-1,1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    return(Rz, Ry)

def create_arrow(scale):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = scale/15
    cylinder_radius = scale/30
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
        cone_height=cone_height,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height)
    return(mesh_frame)

def get_arrow(origin=[0, 0, 0], end=None, vec=None, scale = 1000, color = [0,1,0]):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        #scale = vector_magnitude(vec)
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)

    mesh = create_arrow(scale)
    # Create the arrow
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    mesh.paint_uniform_color(color)
    return(mesh)

def visualizer_3d(vec, type, radius):
    
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius = radius)
    mesh_sphere.paint_uniform_color([0.7,0.1,0.1])

    FOR = get_o3d_FOR(origin = (60,60,0), size = 20)

    if type == 'calyx':
        color = [0,0,1]
    else:
        color = [0,1,0]

    arrow = get_arrow(vec = vec, scale = 100, color = color)

    o3d.visualization.draw_geometries([mesh_sphere, arrow, FOR])
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_shape():
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 2)
    ax.set_ylim(0, 5)
    ax.set_aspect(1)
    
    # Tạo hình theo đúng đường viền mong muốn
    vertices = [
        (0, 0), (1, 0), (1, 3), (0.8, 3), (0.8, 4), (0.2, 4), (0.2, 3), (0, 3), (0, 0)
    ]
    
    shape = patches.Polygon(vertices, closed=True, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(shape)
    
    plt.axis('off')
    plt.show()

draw_shape()

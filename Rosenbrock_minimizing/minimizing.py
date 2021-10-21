import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import numpy as np
from time import time
import argparse


def rosenbrock_funct(x, y):
    return (1-x)**2 + 100*(y-x*x)**2


def gradient(point):
    x = point[0]
    y = point[1]
    dx = 400*x**3 + (-400)*y*x + 2*x - 2
    dy = (-200)*x**2 + 200*y
    return np.array([dx, dy])


def inv_hessian(point):
    x = point[0]
    y = point[1]
    ddx = 1200*x**2 + (-400)*y + 2
    dxy = (-400)*x
    ddy = 200
    hessian = np.array([[ddx, dxy], [dxy, ddy]])
    return np.linalg.inv(hessian)


def draw_plot(points):
    x = np.arange(-5.0, 5.0, 0.1)
    y = np.arange(-5.0, 5.0, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock_funct(X, Y)
    fig = plt.figure(figsize=(8, 8))
    norm = matplotlib.colors.LogNorm(vmin=1, vmax=10000)
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap=plt.cm.jet, norm=norm)
    for pt in points:
        plot_points = matplotlib.patches.Circle(
            (pt[0], pt[1]), 0.04, edgecolor='black')
        ax.add_patch(plot_points)
        art3d.pathpatch_2d_to_3d(plot_points,
                                 z=rosenbrock_funct(pt[0], pt[1]), zdir='z')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.title('$z=(1-x)^2 + 100*(y-x^2)^2$')
    plt.show()


def func_minimum_gradient(start_point, rate=0.00007, iterations=250, epsilo=1e-12):
    start_time = time()
    i = 0
    points = []
    while i < iterations:
        delta = [rate*gradient(start_point)[0], rate*gradient(start_point)[1]]
        x = start_point[0]
        y = start_point[1]
        z = rosenbrock_funct(x, y)
        points.append([x, y, z])
        new_point = [x - delta[0], y - delta[1]]
        eps = ((start_point[0] - new_point[0])**2 +
               (start_point[1] - new_point[1])**2)**(0.5)
        start_point = new_point
        i += 1
        if eps < epsilo:
            break
    func_time = time() - start_time
    draw_plot(points)
    start_point.append(z)
    return start_point, func_time


def func_minimum_newton(start_point, rate=1, iterations=10, epsilo=1e-12):
    start_time = time()
    i = 0
    points = []
    while i < iterations:
        mul_matrix = np.matmul(inv_hessian(start_point), gradient(start_point))
        delta = [rate*mul_matrix.item(0), rate*mul_matrix.item(1)]
        x = start_point[0]
        y = start_point[1]
        z = rosenbrock_funct(x, y)
        points.append([x, y, z])
        new_point = [x - delta[0], y - delta[1]]
        eps = ((start_point[0] - new_point[0])**2 +
               (start_point[1] - new_point[1])**2)**(0.5)
        start_point = new_point
        i += 1
        if eps < epsilo:
            break
    start_point.append(z)
    func_time = time() - start_time
    draw_plot(points)
    return start_point, func_time


def main(args):
    start_point = [args.start_x, args.start_y]
    rate = args.rate
    iter = args.iterations
    if args.auto == "True":
        if args.method == "newton":
            minimum, run_time = func_minimum_newton(start_point)
        elif args.method == "gradient":
            minimum, run_time = func_minimum_gradient(start_point)
        else:
            print("There is no method like that. Use 'newton' or 'gradient'")
            return
    else:
        if rate is None or iter is None:
            print("You have to provide optional arguments.")
            return
        if args.method == "newton":
            minimum, run_time = func_minimum_newton(start_point, rate, iter)
        elif args.method == "gradient":
            minimum, run_time = func_minimum_gradient(start_point, rate, iter)
        else:
            print("There is no method like that. Use 'newton' or 'gradient'")
            return
    print("Minimum in Rosenbrock function:")
    print(f"X: {minimum[0]:.2f}\nY: {minimum[1]:.2f}\nZ: {minimum[2]:.2f}")
    print(f"Run time: {run_time:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method",
                        help="Method of minimizing: gradient or newton")
    parser.add_argument("start_x", type=int,
                        help="X coordinate where function should start")
    parser.add_argument("start_y", type=int,
                        help="Y coordinate where function should start")
    parser.add_argument("auto", choices=["True", "False"],
                        help="True if you want to use defaults arguments")
    parser.add_argument("-r", "--rate", help="Step rate", type=float)
    parser.add_argument("-i", "--iterations", type=int)
    args = parser.parse_args()
    main(args)

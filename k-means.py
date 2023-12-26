from sklearn.datasets import make_blobs
import numpy as np
import random
import matplotlib.pyplot as plt


# Parameters
nb_centers = 5
num_samples = 20000

# Generate data
all_points = make_blobs(n_samples=num_samples, n_features=2, centers=nb_centers, random_state=20)


def visualize_clusters(all_points, centers, title):
    """"
    Visualize all clusters and their center

    Parameters :
        all_points (2D array [[[x_1,y_1], ..., [x_n, y_n]], [center_1, ..., center_n ]]) : List of points [x,y] and list of assigned centers for each point
        centers (2D array) : List of centers [x,y]
        title (string) : Plot title
    """
    points = all_points[0]
    centers = np.array(centers)

    plt.scatter(points[:,0],points[:,1], c=all_points[1], marker='o')
    plt.scatter(centers[:,0],centers[:,1], c=[(1, 0, 0)], marker='X', s=200)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()



def initialize_centers(points, nb_of_k):
    """
    Initialize centers of clusters for K means and K means mini-batch algorithm

    Parameters :
        points (2D array [[x1,y1], ....[xn, yn]]) : List of points [x,y]

    Return : 
        centers (2D array) : List of centers [x,y]
    """

    # List of where centers will be saved
    centers = []

    # Save random int to be sure to have differents numbers
    all_rands = []

    for i in range(nb_of_k):

        rand = random.randint(0, len(points)-1)

        # Make sure there randoms int are differents
        while rand in all_rands:

            rand = random.randint(0, len(points)-1)

        centers.append([points[rand, 0], points[rand, 1]] )
        all_rands.append(rand)

    return centers


def sse_distance(A, B):
    """
    Compute euclidian distance between 2 points

    Parameters:
        A (array [x, y]) : Coordinates of first point
        B (array [x, y]) : Coordinates of second point

    Return :
        eucl_dist (float) : Euclidian distance
    """

    eucl_dist = (B[0] - A[0])**2 + (B[1] - A[1])**2

    return eucl_dist


def find_closest_center(centers, point):
    """
    Find the nearest center to a point

    Parameters :
        centers (2D array) : List of centers [x,y]

    Return :
        index (int) : Index of the nearest center
    """

    # Iterate over all centers
    for i, center in enumerate(centers):
        
        # Compute distance between the point and a center
        dist = sse_distance(center, point)

        # Save the nearest distance
        if i ==0 :
            min_dist = dist
            index = i
        elif dist < min_dist :

            min_dist = dist
            index = i
            
    return index


def compute_clusters(points, centers):
    """
    Find nearest center for each point to update clusters

    Parameters :
        points (2D array [[x1,y1], ....[xn, yn]]) : List of points [x,y]
        centers (2D array) : List of centers [x,y]

    Return :
        cluster_list (2D array [cluster_1, ... , cluster_n]): List of clusters assigned to each point
    """

    clusters_list = []

    # Iterate over all points
    for i in range(len(points)):

        # Find nearest center
        index_cluster = find_closest_center(centers, points[i])

        # Update cluster list
        clusters_list.append(index_cluster)

    clusters_list = np.array(clusters_list)

    return clusters_list


def sse_error(points, clusters, centers_coord):
    """
    Compute sse error for an iteration (sum of euclidian distance between each point and its cluster center)

    Parameters :
        points (2D array [[x1,y1], ....[xn, yn]]) : List of points [x,y]
        clusters (2D array [cluster_1, ... , cluster_n]): List of clusters assigned to each point
        centers_coord (2D array) : List of centers [x,y]
    
    Return : 
        sum_error (float) : sum of euclidian distance between each point and its cluster center
    """

    sum_error = 0.

    # Iterate over all points
    for i in range(len(points)):
        
        # Compute distance
        if clusters[i] < len(centers_coord):
            dist = sse_distance(points[i], centers_coord[clusters[i]])

            sum_error += dist

    return sum_error


def initialize_centers_kmeanspp(points, k):
    """
    Initialize centers for Kmeans ++ algorithm

    Parameters :
        points (2D array [[x1,y1], ....[xn, yn]]) : List of points [x,y]
        k (int) : Number of centers

    Return :
        centers (2D array) : List of centers [x,y]
    """

    centers = []
    probabilities = []

    # Initialize first center
    rand = random.randint(0, len(points[0])-1)
    centers.append([points[rand, 0],points[rand, 1]] )

    # k-1 iterations
    for i in range(k-1):

        sum_probability = 0.
        probabilities = []

        # For each point
        for point in points:

            # Check if current point is a already a center
            if not [point[0], point[1]] in centers:
                
                # Compute distance between the point and its nearest center
                index = find_closest_center(centers, point)
                dist = sse_distance(point, centers[index])

                # Euclidian distance power 4
                dist = dist**2
                
                # Sum probabilities
                # Computed distance is proportionnal to the probability to be chosen
                sum_probability += dist

                # Add probability to array
                probabilities = np.append(probabilities, sum_probability)
                
                # Choose random point 
                # Higher chance for points far from their nearest center
                rand = random.uniform(0., sum_probability)

        # Get choosen centers
        for j in range(len(probabilities)):

            if j == 0 and rand < probabilities[j]:
                centers.append([points[j, 0],points[j, 1]] )

            else:
                if rand > probabilities[j - 1] and rand < probabilities[j]:

                    centers.append([points[j, 0],points[j, 1]] )


    return centers


def visualize_error(iterations, errors, title):
    """
    Visualize error during a whole algorithm
    
    Parameters:
        iterations (int) : Number of iterations
        errors (list) : Error at each iteration
        title (String) : Title for plotting
    """

    plt.scatter([i for i in range(iterations)], errors)
    plt.title(title)
    plt.show()

k_centers = initialize_centers(all_points[0], nb_centers)

def kmeans(points_and_clust, num_of_k, num_it = True):
    """
    Kmeans algorithm

    Parameters:
        points_and_clust (2D array [[[x_1,y_1], ..., [x_n, y_n]], [center_1, ..., center_n ]]) : List of points [x,y] and list of assigned centers for each point
        num_of_k (int) : Number of clusters
        num_it (int | True) : Number of iterations. If not specified, algorithm will stop at convergence (when centers are the same for 2 iterations).

    Return:
        final_centers (2D array) : List of final centers [x,y]

    """

    # # Inititalize k centers
    k_centers = initialize_centers(points_and_clust[0], num_of_k)

    # Attribute centers to each points
    points_clusters = compute_clusters(points_and_clust[0], k_centers)

    # Display clusters at the beginning
    title = "Kmeans : Firsts clusters and centers"
    visualize_clusters(points_and_clust, k_centers, title)

    # For saving history
    history = dict()

    # Save initial values
    error = sse_error(points_and_clust[0], points_clusters, k_centers)
    it_0 = dict(centers = k_centers,
                sse_error = error)
    history[0] = it_0


    new_centers = []
    errors = []
    display = []
    mode = ""

    # Convergence
    if num_it is True:
        print("Kmeans : CONVERGENCE")
        iter = 0
        mode = "CONVERGENCE"

        # For checking convergence
        old_centers = []

        # While not convergence
        while(num_it):

            # Mean of cluster
            for i in range(num_of_k):

                # Index of each k_centers in clusters list
                indexes = [j for j, e in enumerate(points_clusters) if e == i]

                # Select all points points in the current cluster
                selected_points = [points_and_clust[0][idx] for idx in indexes]

                # Mean
                average = sum(selected_points) / len(selected_points)
            
                # List of new centers
                new_centers.append([average[0], average[1]])

            # Find closest centers for each point
            points_clusters = compute_clusters(points_and_clust[0], new_centers)

            # Iteration sse_error
            error = sse_error(points_and_clust[0], points_clusters, new_centers)

            # For visualizing evolution of clusters at each iteration
            current_blobs = [points_and_clust[0], points_clusters]
            title = "Kmeans : Iteration " + str(iter+1)
            display.append([current_blobs, new_centers])

            # Iteration dictionnary
            it = dict(centers = new_centers,
                    sse_error = error)
            
            # Iteration
            iter += 1

            # Update history dictionnary
            history[iter] = it

            # Update errors list
            errors.append(error)

            # If first iteration
            if iter == 1:
                
                # If convergence
                if np.allclose(k_centers, new_centers):

                    # Visualize sse error during algorithm
                    visualize_error(iter, errors, "Kmeans : SSE error during iterations")

                    print("Kmeans : Convergence in ", iter, " iterations")
                    num_it = False
            else :

                # If convergence
                if np.allclose(old_centers, new_centers):

                    # Visualize sse error during algorithm
                    visualize_error(iter, errors, "Kmeans : SSE error during iterations")

                    print("Kmeans : Convergence in ", iter, " iterations")
                    num_it = False
            
            # Update old_center for next iteration
            old_centers = new_centers

            # Reset new centers
            if num_it:
                new_centers = []


    # Specific amount of iterations
    else :
        print("Kmeans : ", num_it ," ITERATIONS")
        mode=str(num_it) + " ITERATIONS"

        for g in range(num_it):

            # Mean of cluster
            for i in range(num_of_k):

                # Index of each k_centers in clusters list
                indexes = [j for j, e in enumerate(points_clusters) if e == i]

                # Select all points points in the current cluster
                selected_points = [points_and_clust[0][idx] for idx in indexes]

                # Mean
                average = sum(selected_points) / len(selected_points)
            
                # List of new centers
                new_centers.append([average[0], average[1]])


            # Find closest centers for each point
            points_clusters = compute_clusters(points_and_clust[0], new_centers)

            # Iteration sse_error
            error = sse_error(points_and_clust[0], points_clusters, new_centers)

            # Visualize evolution of clusters at each iteration
            current_blobs = [points_and_clust[0], points_clusters]
            title = "Kmeans : Iteration " + str(g+1)
            display.append([current_blobs, new_centers])


            # Iteration dictionnary
            it = dict(centers = new_centers,
                    sse_error = error)
            
            # Update history dictionnary
            history[g+1] = it

            # Reset new centers
            if g != num_it -1:
                new_centers = []

            # Update errors list
            errors.append(error)

        # Visualize sse error during algorithm
        visualize_error(num_it, errors, "Kmeans : SSE error during iterations")

    # Visualize evolution during iterations
    for idx, tab in enumerate(display):

        points = tab[0][0]
        centers = np.array(tab[1])

        plt.subplot(len(display) + 1, 2, idx+1)
        plt.scatter(points[:,0],points[:,1], c=tab[0][1], marker='o')
        plt.scatter(centers[:,0],centers[:,1], c=[(1, 0, 0)], marker='X', s=200)
        plt.title("Iteration " + str(idx) )
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # Visualize final results
    final_centers = new_centers
    title = "Kmeans : Finals clusters and centers " + mode
    visualize_clusters(current_blobs, final_centers, title)

    return final_centers


def kmeanspp(points, num_of_k, num_it = True):
    """
    Kmeans ++ algorithm

    Parameters:
        points_and_clust (2D array [[[x_1,y_1], ..., [x_n, y_n]], [center_1, ..., center_n ]]) : List of points [x,y] and list of assigned centers for each point
        num_of_k (int) : Number of clusters
        num_it (int | True) : Number of iterations. If not specified, algorithm will stop at convergence (when centers are the same for 2 iterations).

    Return:
        final_centers (2D array) : List of final centers [x,y]

    """

    # Inititalize k centers
    k_centers = initialize_centers_kmeanspp(points[0], num_of_k)

    # Attribute centers to each points
    points_clusters = compute_clusters(points[0], k_centers)

    # Display clusters at the beginning
    title = " Kmeans ++ : Firsts clusters and centers"
    visualize_clusters(points, k_centers, title)

    # For saving history
    history = dict()

    # Save initial values
    error = sse_error(points[0], points_clusters, k_centers)
    it_0 = dict(centers = k_centers,
                sse_error = error)
    history[0] = it_0


    new_centers = []
    errors = []
    display = []
    mode=""

    # Convergence
    if num_it is True:
        print("Kmeans ++ : CONVERGENCE")
        iter = 0
        mode =  "CONVERGENCE"

        # For checking convergence
        old_centers = []
        
        # While not convergence
        while(num_it):

            # Mean of cluster
            for i in range(num_of_k):

                # Index of each k_centers in clusters list
                indexes = [j for j, e in enumerate(points_clusters) if e == i]

                # Select all points points in the current cluster
                selected_points = [points[0][idx] for idx in indexes]

                # Mean
                average = sum(selected_points) / len(selected_points)
            
                # List of new centers
                new_centers.append([average[0], average[1]])


            # Find closest centers for each point
            points_clusters = compute_clusters(points[0], new_centers)

            # Iteration sse_error
            error = sse_error(points[0], points_clusters, new_centers)

            # For visualizing evolution of clusters at each iteration
            current_blobs = [points[0], points_clusters]
            title = "Iteration " + str(iter+1)
            display.append([current_blobs, new_centers])

            # Iteration dictionnary
            it = dict(centers = new_centers,
                    sse_error = error)
            
            # Iteration
            iter += 1

            # Update history dictionnary
            history[iter] = it

            # Update errors list
            errors.append(error)

            
            # If first iteration
            if iter == 1:
                
                # If convergence
                if np.allclose(k_centers, new_centers):

                    visualize_error(iter, errors, "Kmeans ++ : SSE error during iterations")

                    print("Kmeans ++ : Convergence in ", iter, " iterations")
                    num_it = False
            else :

                # If convergence
                if np.allclose(old_centers, new_centers):

                    visualize_error(iter, errors, "Kmeans ++ : SSE error during iterations")

                    print("Kmeans ++ : Convergence in ", iter, " iterations")
                    num_it = False
            
            # Update old_center for next iteration
            old_centers = new_centers

            # Reset new centers
            if num_it:
                new_centers = []


    # Specific amount of iterations
    else :
        print("Kmeans ++ : ",num_it, " ITERATIONS")
        mode = str(num_it) + " ITERATIONS"

        for g in range(num_it):

            # Mean of cluster
            for i in range(num_of_k):

                # Index of each k_centers in clusters list
                indexes = [j for j, e in enumerate(points_clusters) if e == i]

                # Select all points points in the current cluster
                selected_points = [points[0][idx] for idx in indexes]

                # Mean
                average = sum(selected_points) / len(selected_points)
            
                # List of new centers
                new_centers.append([average[0], average[1]])


            # Find closest centers for each point
            points_clusters = compute_clusters(points[0], new_centers)

            # Iteration sse_error
            error = sse_error(points[0], points_clusters, new_centers)

            # For visualizing evolution of clusters at each iteration
            current_blobs = [points[0], points_clusters]
            title = "Kmeans ++ : Iteration " + str(g+1)
            display.append([current_blobs, new_centers])

            # Iteration dictionnary
            it = dict(centers = new_centers,
                    sse_error = error)
            
            # Update history dictionnary
            history[g+1] = it

            # Reset new centers
            if g != num_it -1:
                new_centers = []

            # Update errors list
            errors.append(error)

        # Visualize sse error during algorithm
        visualize_error(num_it, errors, " Kmeans ++ : SSE error during iterations")

    # Visualize evolution during iterations
    for idx, tab in enumerate(display):

        points = tab[0][0]
        centers = np.array(tab[1])

        plt.subplot((len(display) // 2) + 1, 2, idx+1)
        plt.scatter(points[:,0],points[:,1], c=tab[0][1], marker='o')
        plt.scatter(centers[:,0],centers[:,1], c=[(1, 0, 0)], marker='X', s=200)
        plt.title("KPP: Iteration " + str(idx) )
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # Visualize final results
    final_centers = new_centers
    title = "Finals clusters and centers Kmeans ++ " + mode
    visualize_clusters(current_blobs, final_centers, title)

    return final_centers


def kmeans_mini_batch(points, num_of_k:int, batch_size: int , num_it: int):
    """
    Kmeans mini batch algorithm

    Parameters:
        points_and_clust (2D array [[[x_1,y_1], ..., [x_n, y_n]], [center_1, ..., center_n ]]) : List of points [x,y] and list of assigned centers for each point
        num_of_k (int) : Number of clusters
        batch_size (int) : Number of points per batch
        num_it (int) : Number of iterations

    Return:
        final_centers (2D array) : List of final centers [x,y]

    """
    # Inititalize k centers
    k_centers = initialize_centers(points[0], num_of_k)

    # List of assigned centers
    # Initialized with incorrect value (some points will never be choosen during algorithm)
    assigned_centers = [num_of_k * 2] * len(points[0])

    # Display clusters at the beginning
    title = "Kmeans mini-batchs : Randomly initialized centers"
    visualize_clusters([points[0], assigned_centers], k_centers, title)

    # List of new centers at each iterations
    new_centers = []

    # History dict
    history = dict()

    # List of sse_errors during iterations for plotting
    errors = []
    
    # Algorithm
    print("Kmeans mini-batchs ", num_it, " ITERATIONS")
    for it in range(num_it):

        # Reset new centers list
        new_centers = []

        # Select random indexes for a batch
        selected_points = random.sample(range(len(points[0])), batch_size)

        # Get coordinates of selected points
        current_points = [points[0][idx] for idx in selected_points]

        # Find closest center for each point
        points_clusters_assigned = compute_clusters(current_points, k_centers)

        # Iterate over center
        for index_center in range(num_of_k):

            # Indexes of each centers in clusters list
            indexes = [idx for idx, center in enumerate(points_clusters_assigned) if center == index_center]
            
            # Get all points assigned to that center
            points_assigned_to_center = [current_points[idx] for idx in indexes]

            # Mean of all assigned points
            average = sum(points_assigned_to_center) / len(points_assigned_to_center)

            # Compute Learning rate
            learning_rate = 1 / len(indexes)

            # Compute new center
            new_center = (1. - learning_rate) * np.array(k_centers[index_center], dtype=np.float64 ) + learning_rate * average
            new_centers.append([new_center[0], new_center[1]])

            # Update assigned centers
            for idx in indexes:
                for all_idx, real_index in enumerate(selected_points):

                    if idx == all_idx:

                        assigned_centers[real_index] = index_center


        # Iteration sse_error
        error = sse_error(points[0], assigned_centers, new_centers)
        
        # Update centers list
        k_centers = new_centers

        # Update history
        history[it+1] = dict(centers=new_centers, sse_error=error, already_selected_points=assigned_centers)

        # Visualize itérations
        current_data = [points[0], assigned_centers]
        visualize_clusters(current_data, new_centers, "Kmeans-mini_batchs : Iteration " + str(it+1))

        # Visualize errors
        if it == num_it - 1:
            errors.append(error)

            plt.scatter([num_it], errors)
            plt.title("SSE error during iterations of Kmeans mini-batchs")
            plt.show()

    # Visualize results
    current_data = [points[0], assigned_centers]
    visualize_clusters(current_data, new_centers, "Kmeans-mini_batchs : END")

    return k_centers



# K-means
# kmeans(all_points, nb_centers)
# kmeans(all_points, nb_centers,10)

# K-means ++
# kmeanspp(all_points, nb_centers)
# kmeanspp(all_points, nb_centers,10)

# K-means mini batch
# kmeans_mini_batch(all_points, nb_centers, len(all_points[0]) // nb_centers, 10)





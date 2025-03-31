import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;


/**
 * KMeansClusterer.java - a JUnit-testable interface for the Model AI Assignments k-Means Clustering exercises.
 * @author Todd W. Neller
 */
public class KMeansClusterer3 {
	private int dim; // the number of dimensions in the data
	private int k, kMin, kMax; // the allowable range of the of clusters
	private int iter; // the number of k-Means Clustering iterations per k
	private double[][] data; // the data vectors for clustering
	private double[][] centroids; // the cluster centroids
	private int[] clusters; // assigned clusters for each data point
	private Random random = new Random();


	/**
	 * Read the specified data input format from the given file and return a double[][] with each row being a data point and each column being a dimension of the data.
	 * @param filename the data input source file
	 * @return a double[][] with each row being a data point and each column being a dimension of the data
	 */
	public double[][] readData(String filename) {
		int numPoints = 0;
		
		try{
			Scanner in = new Scanner(new File(filename));
			try {
				dim = Integer.parseInt(in.nextLine().split(" ")[1]);
				numPoints = Integer.parseInt(in.nextLine().split(" ")[1]);
			}
			catch (Exception e) {
				System.err.println("Invalid data file format. Exiting.");
				e.printStackTrace();
				System.exit(1);
			}
			double[][] data = new double[numPoints][dim];
			for (int i = 0; i < numPoints; i++) {
				String line = in.nextLine();
				Scanner lineIn = new Scanner(line);
				for (int j = 0; j < dim; j++)
					data[i][j] = lineIn.nextDouble();
				lineIn.close();
			}
			in.close();
			return data;
		}
		catch (FileNotFoundException e){
			System.err.println("Could not locate source file. Exiting.");
			e.printStackTrace();
			System.exit(1);
		}		
		return null;
	}

	/**
	 * Set the given data as the clustering data as a double[][] with each row being a data point and each column being a dimension of the data.
	 * @param data the given clustering data
	 */
	public void setData(double[][] data) {
		this.data = data;		
		this.dim = data[0].length;
	}

	/**
	 * Return the clustering data as a double[][] with each row being a data point and each column being a dimension of the data.
	 * @return the clustering data
	 */
	public double[][] getData() {
		return data;
	}

	/**
	 * Return the number of dimensions of the clustering data.
	 * @return the number of dimensions of the clustering data
	 */
	public int getDim() {
		return dim;
	}

	/**
	 * Set the minimum and maximum allowable number of clusters k.  If a single given k is to be used, then kMin == kMax.  If kMin &lt; kMax, then all k from kMin to kMax inclusive will be
	 * compared using the gap statistic.  The minimum WCSS run of the k with the maximum gap will be the result.
	 * @param kMin minimum number of clusters
	 * @param kMax maximum number of clusters
	 */
	public void setKRange(int kMin, int kMax) {
		this.kMin = kMin; 
		this.kMax = kMax;
		this.k = kMin;
	}
	
	/**
	 * Return the number of clusters k.  After calling kMeansCluster() with a range from kMin to kMax, this value will be the k yielding the maximum gap statistic.
	 * @return the number of clusters k.
	 */
	public int getK() {
		return k;
	}
	
	/**
	 * Set the number of iterations to perform k-Means Clustering and choose the minimum WCSS result.
	 * @param iter the number of iterations to perform k-Means Clustering
	 */
	public void setIter(int iter) {
		this.iter = iter;	
	}

	/**
	 * Return the array of centroids indexed by cluster number and centroid dimension.
	 * @return the array of centroids indexed by cluster number and centroid dimension.
	 */
	public double[][] getCentroids() {
		return centroids;
	}

	/**
	 * Return a parallel array of cluster assignments such that data[i] belongs to the cluster clusters[i] with centroid centroids[clusters[i]].
	 * @return a parallel array of cluster assignments
	 */
	public int[] getClusters() {
		return clusters;
	}

	/**
	 * Return the Euclidean distance between the two given point vectors.
	 * @param p1 point vector 1
	 * @param p2 point vector 2
	 * @return the Euclidean distance between the two given point vectors
	 */
	private double getDistance(double[] p1, double[] p2) {
		double sumOfSquareDiffs = 0;
		for (int i = 0; i < p1.length; i++) {
			double diff = p1[i] - p2[i];
			sumOfSquareDiffs += diff * diff;
		}
		return Math.sqrt(sumOfSquareDiffs);
	}
	
	/**
	 * Return the minimum Within-Clusters Sum-of-Squares measure for the chosen k number of clusters.
	 * @return the minimum Within-Clusters Sum-of-Squares measure
	 */
	public double getWCSS() {
        double wcss = 0.0;
        for (int i = 0; i < data.length; i++) {
            int cluster = clusters[i];
            wcss += Math.pow(getDistance(data[i], centroids[cluster]), 2);
        }
        return wcss;	
    }

	/**
	 * Assign each data point to the nearest centroid and return whether or not any cluster assignments changed.
	 * @return whether or not any cluster assignments changed
	 */
	public boolean assignNewClusters() {
        boolean changed = false;
        for (int i = 0; i < data.length; i++) {
            int closest = 0;
            double minDist = getDistance(data[i], centroids[0]);

            for (int j = 1; j < centroids.length; j++) {
                double dist = getDistance(data[i], centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    closest = j;
                }
            }

            if (clusters[i] != closest) {
                clusters[i] = closest;
                changed = true;
            }
        }
        return changed;
	}
	
	/**
	 * Compute new centroids at the mean point of each cluster of points.
	 */
	public void computeNewCentroids() {
        int[] counts = new int[k];
        centroids = new double[k][dim];

        for (int i = 0; i < data.length; i++) {
            int cluster = clusters[i];
            counts[cluster]++;
            for (int j = 0; j < dim; j++) {
                centroids[cluster][j] += data[i][j];
            }
        }

        for (int i = 0; i < k; i++) {
            if (counts[i] > 0) {
                for (int j = 0; j < dim; j++) {
                    centroids[i][j] /= counts[i];
                }
            }
        }
    }

	
	/**
	 * Perform k-means clustering with Forgy initialization and return the 0-based cluster assignments for corresponding data points.
	 * If iter &gt; 1, choose the clustering that minimizes the WCSS measure.
	 * If kMin &lt; kMax, select the k maximizing the gap statistic using 100 uniform samples uniformly across given data ranges.
	 */
	public void kMeansCluster() {
        double maxGap = Double.NEGATIVE_INFINITY;
        int bestK = kMin;
        double[][] bestCentroids = null;
        int[] bestClusters = null;

        double[] minVals = new double[dim];
        double[] maxVals = new double[dim];

        for (int d = 0; d < dim; d++) {
            minVals[d] = Double.MAX_VALUE;
            maxVals[d] = Double.MIN_VALUE;
            for (int i = 0; i < data.length; i++) {
                if (data[i][d] < minVals[d]) minVals[d] = data[i][d];
                if (data[i][d] > maxVals[d]) maxVals[d] = data[i][d];
            }
        }

        for (int currentK = kMin; currentK <= kMax; currentK++) {
            double minWCSS = Double.MAX_VALUE;
            double logMinWCSS = 0;

            for (int i = 0; i < iter; i++) {
                clusters = new int[data.length];
                centroids = new double[currentK][dim];

                for (int j = 0; j < currentK; j++) {
                    int randomIndex = random.nextInt(data.length);
                    centroids[j] = Arrays.copyOf(data[randomIndex], dim);
                }

                boolean changed;
                int iterations = 0;

                do {
                    changed = assignNewClusters();
                    computeNewCentroids();
                    iterations++;
                } while (changed && iterations < 100);

                double currentWCSS = getWCSS();

                if (currentWCSS < minWCSS) {
                    minWCSS = currentWCSS;
                    logMinWCSS = Math.log(minWCSS);
                    bestClusters = Arrays.copyOf(clusters, clusters.length);
                    bestCentroids = Arrays.copyOf(centroids, centroids.length);
                    bestK = currentK;
                }
            }

            double avgRandWCSS = 0;
            for (int i = 0; i < 100; i++) {
                double[][] randomData = new double[data.length][dim];
                for (int j = 0; j < data.length; j++) {
                    for (int d = 0; d < dim; d++) {
                        randomData[j][d] = minVals[d] + random.nextDouble() * (maxVals[d] - minVals[d]);
                    }
                }
                this.setData(randomData);
                this.k = currentK;
                this.kMeansCluster();
                avgRandWCSS += Math.log(this.getWCSS());
            }

            avgRandWCSS /= 100.0;
            double gapK = avgRandWCSS - logMinWCSS;

            if (gapK > maxGap) {
                maxGap = gapK;
                clusters = bestClusters;
                centroids = bestCentroids;
                k = bestK;
            }
        }
        System.out.println("Best k: " + k + ", Max Gap: " + maxGap);
	}

	/**
	 * Read UNIX-style command line parameters to as to specify the type of k-Means Clustering algorithm applied to the formatted data specified.
	 * "-k int" specifies both the minimum and maximum number of clusters. "-kmin int" specifies the minimum number of clusters. "-kmax int" specifies the maximum number of clusters. 
	 * "-iter int" specifies the number of times k-Means Clustering is performed in iteration to find a lower local minimum.
	 * "-in filename" specifies the source file for input data. "-out filename" specifies the destination file for cluster data.
	 * @param args command-line parameters specifying the type of k-Means Clustering
	 */
	public static void main(String[] args) {
		int kMin = 2, kMax = 2, iter = 1;
		ArrayList<String> attributes = new ArrayList<String>();
		ArrayList<Integer> values = new ArrayList<Integer>();
		int i = 0;
		String infile = null;
		String outfile = null;
		while (i < args.length) {
			if (args[i].equals("-k") || args[i].equals("-kmin") || args[i].equals("-kmax") || args[i].equals("-iter")) {
				attributes.add(args[i++].substring(1));
				if (i == args.length) {
					System.err.println("No integer value for" +  attributes.get(attributes.size() - 1) + ".");
					System.exit(1);
				}
				try {
					values.add(Integer.parseInt(args[i]));
					i++;
				}
				catch (Exception e) {
					System.err.printf("Error parsing \"%s\" as an integer.", args[i]);
					System.exit(2);
				}
			}
			else if(args[i].equals("-in")){
				i++;
				if (i == args.length){
					System.err.println("No string value provided for input source");
					System.exit(1);
				}
				infile = args[i];
				i++;
			}
			else if(args[i].equals("-out")){
				i++;
				if (i == args.length){
					System.err.println("No string value provided for output source");
					System.exit(1);
				}
				outfile = args[i];
				i++;
			}
		}

		for (i = 0; i < attributes.size(); i++) {
			String attribute = attributes.get(i);
			if (attribute.equals("k"))
				kMin = kMax = values.get(i);
			else if (attribute.equals("kmin"))
				kMin = values.get(i);
			else if (attribute.equals("kmax"))
				kMax = values.get(i);
			else if (attribute.equals("iter"))
				iter = values.get(i);
		}
		
		KMeansClusterer km = new KMeansClusterer();
		km.setKRange(kMin, kMax);
		km.setIter(iter);
		km.setData(km.readData(infile));
		km.kMeansCluster();
		km.writeClusterData(outfile);
	}
}

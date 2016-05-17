import evaluationMetric.Container;
import online.OnlineParser;
import org.grouplens.lenskit.vectors.SparseVector;
import util.*;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;

public class PreparationRunner {
	public static void main(String args[]) {
		//PrepareUtil.prepareSmallDataset("D:\\bigdata\\movielens\\ml-100k\\u.item");
		//PrepareUtil.prepareBigDataset("D:\\bigdata\\movielens\\hetrec\\movie_genres.dat");
		//PrepareUtil.prepareYahooDataset("D:\\bigdata\\Yahoo Movies\\ratings.txt");
		//PrepareUtil.preprocess10M("dataset/ml/10m/ratings.dat", "dataset/ml/10m/content.dat");
		//PrepareUtil.prepareContentYahooDataset("D:\\bigdata\\Yahoo Movies\\movie_db_yoda");
		//PrepareUtil.printUserItemRatingNumber("dataset/ml/small/ratings_unpop.dat");
		//PrepareUtil.printDissimilarityRating("dataset/ml/small/ratings.dat", "dataset/ml/small/content.dat");
		//PrepareUtil.printUnpopularityRating("dataset/ml/big/ratings.dat");
		//PrepareUtil.printItemPopularity("dataset/ml/big/ratings.dat");//"dataset/ml/big/ratings_unpop.dat");
		//PrepareUtil.generateUnpopDataset("dataset/ml/big/ratings.dat", 344);
		//printMovies();
		//printMetrics();
		//PrepareUtil.printGenresNumber("dataset/yahoo/ratings.dat", "dataset/yahoo/content.dat");
		//PrepareUtil.ratingsToGenres("dataset/ml/small/ratings.dat", "dataset/ml/small/content.dat");
		PrepareUtil.ratingNumber("dataset/ml/big/ratings.dat");
		//process();
		//PrepareUtil.getPopMap("dataset/ml/big/ratings.dat");
	}

	private static void process() {
		//OnlineParser.processInputData("D:\\gdrive\\PhD stuff\\Research\\Serendipitous algorithm\\results\\OnlineExperiments\\Ratings");
		//OnlineParser.processUserInputData("D:\\gdrive\\PhD stuff\\Research\\Serendipitous algorithm\\results\\OnlineExperiments\\Den.csv", "100040");
		//OnlineParser.cleanRecommendations("D:\\gdrive\\PhD stuff\\Research\\Serendipitous algorithm\\results\\OnlineExperiments\\res.txt");
		OnlineParser.generateRecommendationList("D:\\gdrive\\PhD stuff\\Research\\Serendipitous algorithm\\results\\OnlineExperiments\\cleanRecommendations.txt", "D:\\gdrive\\PhD stuff\\Research\\Serendipitous algorithm\\results\\OnlineExperiments");
		//OnlineParser.removeFunc("D:\\bigdata\\movielens\\hetrec\\extension\\Test\\ub.pred");
		/*String num = "100006";
		OnlineParser.generateFinalResults("D:\\gdrive\\PhD stuff\\Research\\Serendipitous algorithm\\results\\OnlineExperiments\\Recommendations\\result\\" + num + ".txt",
				//"D:\\gdrive\\PhD stuff\\Research\\Serendipitous algorithm\\results\\OnlineExperiments\\Recommendations\\Personalized\\cleanRecommendations.txt",
				"D:\\gdrive\\PhD stuff\\Research\\Serendipitous algorithm\\results\\OnlineExperiments\\Recommendations\\Random\\cleanRecommendations.txt",
				"dataset/ml/big/ratingsExtended.dat", num);*/
		//OnlineParser.printStatistics("D:\\bigdata\\movielens\\hetrec\\movies.dat");
		//OnlineParser.printNumberOfMoviesReleasedAfter("D:\\bigdata\\movielens\\hetrec\\movies.dat", 1980);
		//OnlineParser.checkTags("D:\\bigdata\\movielens\\hetrec\\movie_tags.dat");
	}

	private static void printMovies() {
		String[] array = {"356", "318", "1270", "7254", "39446", "53000", "46653", "2692", "2959", "47200", "3052", "344", "31696", "5903", "8810", "7196", "480", "589", "1258", "1721", "1193", "1407", "1200", "44", "588", "1", "586", "4470", "4886", "4388", "4992", "1253"};
		Map<String, String> map = new HashMap<String, String>();
		for (String id : array) {
			map.put(id, "");
		}
		PrepareUtil.printMoviesAndGetNames("D:\\bigdata\\movielens\\hetrec\\movies.dat", map, "\t");
		//PrepareUtil.getMoviesByName("D:\\bigdata\\movielens\\ml-1m\\movies.dat", map);
		System.out.println();
		for (String id : array) {
			System.out.println(id + "\t" + map.get(id));
		}
	}

	private static void printMetrics() {
		Util.setParameters();
		String[] profile = {"356", "318", "1270", "7254", "39446", "53000", "46653", "2692", "2959", "47200", "3052", "344", "31696", "5903", "8810", "7196", "480", "589", "1258", "1721", "1193", "1407", "1200", "44", "588", "1", "586", "4470", "4886", "4388", "4992", "1253"};
		String[] allItems = {"2571", "4993", "296", "5952", "2858", "7153", "593", "4306", "2762", "260", "858", "50", "44555", "3435", "296", "1203", "1221", "750", "912", "6016", "26729", "5385", "38304", "37240", "1189", "3677", "1797", "7836", "27878", "1111", "8891", "61210", "273", "6569", "1386", "51705", "3775", "7959", "4250", "2504", "2571", "296", "2858", "593", "260", "1196", "2762", "1198", "50", "4993", "50", "1196", "593", "1198", "47", "260", "32", "527", "1617", "1136", "296", "2571", "5952", "7153", "2858", "593", "4306", "4226", "50", "260"};
		Double[] allRatings = {3.5d, 2d, 3d, 3d, 3.5d, 3d, 4d, 3d, 4d, 2d, 3d, 3.5d, 2.5d, 1d, 3d, 2d, 2d, 1d, 1d, 3d, 2d, 2d, 2d, 3.5d, 2.5d, 3.5d, 2d, 1d, 3d, 2.5d, 3d, 2d, 2d, 2d, 2d, 3d, 1d, 3d, 3.5d, 2d, 3.5d, 3d, 3.5d, 4d, 2d, 2d, 4d, 3d, 3.5d, 2d, 3.5d, 2d, 4d, 3d, 1d, 2d, 3d, 3d, 3.5d, 2.5d, 3d, 3.5d, 3d, 3d, 3.5d, 4d, 3d, 4d, 3.5d, 2d};
		String[] items = {"4226", "1196", "1198", "260", "4306", "4993", "2762", "3677", "5385", "3435", "2858", "3775", "1221", "27878", "1386", "4250", "26729", "8891", "51705", "2504", "1189", "44555", "7836", "37240", "6569", "2571", "750", "1136", "47", "1111", "296", "7153", "1797", "7959", "858", "912", "273", "1617", "1203", "32", "5952", "6016", "38304", "61210", "527", "593", "50"};
		Map<String, Double> groundTruthMap = getRatingMap(allItems, allRatings);
		ContentAverageDissimilarity.create("dataset/ml/big/content.dat");
		Map<String, Integer> popMap = PrepareUtil.getPopMap("dataset/ml/big/ratings.dat");
		int primTheshold = getPrimitiveThreshold(popMap);
		double max = getMax(popMap);
		double uSum = 0;
		double dSum = 0;
		double tsSum = 0;
		System.out.println("id\tU\tD\tSer\tRDU");
		for (int i = 0; i < items.length; i++) {
			double unpop = 1 - (double) popMap.get(items[i]) / max;
			uSum += unpop;
			double dissimilarity = getDissimilarity(profile, items[i]);
			dSum += dissimilarity;
			double tSer = getTraditionalSerendipity(popMap, items[i], primTheshold, groundTruthMap.get(items[i]));
			tsSum += tSer;
			double ser = getSerendipityVal(groundTruthMap.get(items[i]), dissimilarity, unpop);
			System.out.println(items[i] + "\t" + unpop + "\t" + dissimilarity + "\t" + tSer + "\t" + ser);
		}
		uSum /= items.length;
		dSum /= items.length;
		tsSum /= items.length;
		System.out.println("Total\t" + uSum + "\t" + dSum + "\t" + tsSum);
		System.out.println("NRDU" + "\t" + getNRDU(items, groundTruthMap, profile, popMap, max) + "\tNDCG\t" + getNDCG(items, groundTruthMap));
	}

	private static Map<String, Double> getRatingMap(String[] allItems, Double[] allRatings) {
		Map<String, Double> map = new HashMap<String, Double>();
		for (int i = 0; i < allItems.length; i++) {
			map.put(allItems[i], allRatings[i]);
		}
		return map;
	}

	private static double getNRDU(String[] items, Map<String, Double> map, String[] profile, Map<String, Integer> popMap, double maxPop) {
		List<Container<Double>> containers = new ArrayList<Container<Double>>();
		for (int i = 0; i < items.length; i++) {
			double dissimilarity = getDissimilarity(profile, items[i]);
			double unpop = 1 - (double) popMap.get(items[i]) / maxPop;
			containers.add(new Container<Double>(Long.valueOf(items[i]), getSerendipityVal(map.get(items[i]), dissimilarity, unpop)));
		}
		double actualDCG = getDCG(containers);
		containers = new ArrayList<Container<Double>>();
		for (Map.Entry<String, Double> entry : map.entrySet()) {
			double dissimilarity = getDissimilarity(profile, entry.getKey());
			double unpop = 1 - (double) popMap.get(entry.getKey()) / maxPop;
			containers.add(new Container<Double>(Long.valueOf(entry.getKey()), getSerendipityVal(entry.getValue(), dissimilarity, unpop)));
		}
		Collections.sort(containers);
		Collections.reverse(containers);
		double perfectDCG = getDCG(containers);
		return actualDCG / perfectDCG;
	}

	private static double getNDCG(String[] items, Map<String, Double> map) {
		List<Container<Double>> containers = new ArrayList<Container<Double>>();
		for (int i = 0; i < items.length; i++) {
			containers.add(new Container<Double>(Long.valueOf(items[i]), map.get(items[i])));
		}
		double actualDCG = getDCG(containers);
		containers = new ArrayList<Container<Double>>();
		for (Map.Entry<String, Double> entry : map.entrySet()) {
			containers.add(new Container<Double>(Long.valueOf(entry.getKey()), entry.getValue()));
		}
		Collections.sort(containers);
		Collections.reverse(containers);
		double perfectDCG = getDCG(containers);
		return actualDCG / perfectDCG;
	}

	private static double getDCG(List<Container<Double>> list) {
		double dcg = list.get(0).getValue();
		for (int i = 1; i < list.size(); i++) {
			dcg += list.get(i).getValue() * Math.log(2) / Math.log(i + 1);
		}
		return dcg;
	}

	private static double getSerendipityVal(double r, double d, double u) {
		if (r <= Settings.R_THRESHOLD) {
			return 0;
		}
		if (d <= Settings.D_THRESHOLD) {
			return 0;
		}
		if (u <= Settings.U_THRESHOLD) {
			return 0;
		}
		return r / Settings.MAX + d + u;
	}

	private static int getTraditionalSerendipity(Map<String, Integer> popMap, String id, int threshold, double rating) {
		int pop = popMap.get(id);
		if (pop >= threshold) {
			return 0;
		}
		if (rating <= Settings.R_THRESHOLD) {
			return 0;
		}
		return 1;
	}

	private static int getPrimitiveThreshold(Map<String, Integer> popMap) {
		List<Container<Integer>> containers = new ArrayList<Container<Integer>>();
		for (Map.Entry<String, Integer> entry : popMap.entrySet()) {
			containers.add(new Container<Integer>(Long.valueOf(entry.getKey()), entry.getValue()));
		}
		Collections.sort(containers);
		Collections.reverse(containers);
		return containers.get(Settings.POPULAR_ITEMS_SERENDIPITY_NUMBER).getValue();
	}

	private static double getDissimilarity(String[] profile, String itemId) {
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> map = dissimilarity.getItemContentMap();
		Long id = Long.valueOf(itemId);
		SparseVector vec = map.get(id);
		double dissim = 0;
		for (String ratedItem : profile) {
			Long ratedId = Long.valueOf(ratedItem);
			SparseVector ratedVec = map.get(ratedId);
			dissim += 1 - ContentUtil.getJaccard(ratedVec, vec);
		}
		return dissim / profile.length;
	}

	private static double getMax(Map<String, Integer> popMap) {
		double max = 0;
		for (Integer val : popMap.values()) {
			max = Math.max(val, max);
		}
		return max;
	}

}

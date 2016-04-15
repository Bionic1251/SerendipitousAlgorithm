package util;

public class Settings {
	public static int CROSSFOLD_NUMBER;
	public static int HOLDOUT_NUMBER;
	public static int POPULAR_ITEMS_SERENDIPITY_NUMBER;
	public static int RANDOM_ITEMS_FOR_CANDIDATES;
	public static int POPULAR_ITEMS_FOR_CANDIDATES;
	public static String DATASET = "ml/small/ratings.dat";
	public static String DATASET_CONTENT = "ml/small/content.dat";
	public static String TRAIN_TEST_FOLDER_NAME = "task";
	public static String OUTPUT_PATH = "/out.csv";
	public static String OUTPUT_USER_PATH = "/user.csv";
	public static String OUTPUT_ITEM_PATH = "/item.csv";

	public static double MIN;
	public static double MAX;

	public static double R_THRESHOLD;
	public static double D_THRESHOLD;
	public static double U_THRESHOLD;
	public static double ALPHA;
	public static int ITERATION_COUNT_SVD;
	public static int ITERATION_COUNT_SPR;
	public static int ITERATION_COUNT_PURE_SVD;
	public static int FEATURE_COUNT;
	public static double LEARNING_RATE;
	public static double REGULARIZATION_TERM;

	public static double ZHENG_LEARNING_RATE;
	public static double ZHENG_REGULARIZATION_TERM;

	public static double LU_LEARNING_RATE;
	public static double LU_REGULARIZATION_TERM;

	public static double FRACTION;
	public static double GENRES_NUMBER;
}

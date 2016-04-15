package util;

import lc.Normalizer;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

public class Util {
	public static Normalizer getVectorNormalizer(SparseVector ratings) {
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		for (VectorEntry e : ratings.view(VectorEntry.State.EITHER)) {
			min = Math.min(min, e.getValue());
			max = Math.max(max, e.getValue());
		}
		return new Normalizer(min, max);
	}

	public static Normalizer getMapNormalizer(Map<Long, Double> map) {
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		for (double val : map.values()) {
			min = Math.min(min, val);
			max = Math.max(max, val);
		}
		return new Normalizer(min, max);
	}

	public static void setParameters() {
		Properties prop = new Properties();
		InputStream input = null;

		try {
			input = new FileInputStream("config.properties");
			prop.load(input);

			Settings.DATASET = (String) prop.get("dataset");
			System.out.println("dataset " + Settings.DATASET);

			Settings.DATASET_CONTENT = (String) prop.get("dataset_content");
			System.out.println("dataset_content " + Settings.DATASET_CONTENT);

			Settings.MIN = Double.valueOf((String) prop.get("min_rating"));
			System.out.println("min_rating " + Settings.MIN);

			Settings.MAX = Double.valueOf((String) prop.get("max_rating"));
			System.out.println("max_rating " + Settings.MAX);

			Settings.CROSSFOLD_NUMBER = Integer.valueOf((String) prop.get("crossfold"));
			System.out.println("crossfold " + Settings.CROSSFOLD_NUMBER);

			Settings.HOLDOUT_NUMBER = Integer.valueOf((String) prop.get("holdout"));
			System.out.println("holdout " + Settings.HOLDOUT_NUMBER);

			Settings.POPULAR_ITEMS_SERENDIPITY_NUMBER = Integer.valueOf((String) prop.get("popular_items_number"));
			System.out.println("popular_items_number " + Settings.POPULAR_ITEMS_SERENDIPITY_NUMBER);

			Settings.RANDOM_ITEMS_FOR_CANDIDATES = Integer.valueOf((String) prop.get("random_items_candidates"));
			System.out.println("random_items_candidates " + Settings.RANDOM_ITEMS_FOR_CANDIDATES);

			Settings.POPULAR_ITEMS_FOR_CANDIDATES = Integer.valueOf((String) prop.get("popular_items_candidates"));
			System.out.println("popular_items_candidates " + Settings.POPULAR_ITEMS_FOR_CANDIDATES);

			Settings.TRAIN_TEST_FOLDER_NAME = (String) prop.get("train_folder");
			System.out.println("train_folder " + Settings.TRAIN_TEST_FOLDER_NAME);

			Settings.OUTPUT_PATH = (String) prop.get("output");
			System.out.println("output " + Settings.OUTPUT_PATH);

			Settings.OUTPUT_USER_PATH = (String) prop.get("output_user");
			System.out.println("output_user " + Settings.OUTPUT_USER_PATH);

			Settings.OUTPUT_ITEM_PATH = (String) prop.get("outout_item");
			System.out.println("outout_item " + Settings.OUTPUT_ITEM_PATH);

			Settings.R_THRESHOLD = Double.valueOf((String) prop.get("r_threshold"));
			System.out.println("r_threshold " + Settings.R_THRESHOLD);

			Settings.D_THRESHOLD = Double.valueOf((String) prop.get("d_threshold"));
			System.out.println("d_threshold " + Settings.D_THRESHOLD);

			Settings.U_THRESHOLD = Double.valueOf((String) prop.get("u_threshold"));
			System.out.println("u_threshold " + Settings.U_THRESHOLD);

			Settings.FEATURE_COUNT = Integer.valueOf((String) prop.get("feature_count"));
			System.out.println("feature_count " + Settings.FEATURE_COUNT);

			Settings.ITERATION_COUNT_SVD = Integer.valueOf((String) prop.get("iteration_count_svd"));
			System.out.println("iteration_count_svd " + Settings.ITERATION_COUNT_SVD);

			Settings.ITERATION_COUNT_PURE_SVD = Integer.valueOf((String) prop.get("iteration_count_pure_svd"));
			System.out.println("iteration_count_pure_svd " + Settings.ITERATION_COUNT_PURE_SVD);

			Settings.ITERATION_COUNT_SPR = Integer.valueOf((String) prop.get("iteration_count_spr"));
			System.out.println("iteration_count_spr " + Settings.ITERATION_COUNT_SPR);

			Settings.LEARNING_RATE = Double.valueOf((String) prop.get("learning_rate"));
			System.out.println("learning_rate " + Settings.LEARNING_RATE);

			Settings.REGULARIZATION_TERM = Double.valueOf((String) prop.get("regularization_term"));
			System.out.println("regularization_term " + Settings.REGULARIZATION_TERM);

			Settings.ZHENG_LEARNING_RATE = Double.valueOf((String) prop.get("zheng_learning_rate"));
			System.out.println("zheng_learning_rate " + Settings.ZHENG_LEARNING_RATE);

			Settings.ZHENG_REGULARIZATION_TERM = Double.valueOf((String) prop.get("zheng_regularization_term"));
			System.out.println("zheng_regularization_term " + Settings.ZHENG_REGULARIZATION_TERM);

			Settings.ALPHA = Double.valueOf((String) prop.get("alpha"));
			System.out.println("alpha " + Settings.ALPHA);

			Settings.LU_LEARNING_RATE = Double.valueOf((String) prop.get("lu_learning_rate"));
			System.out.println("lu_learning_rate " + Settings.LU_LEARNING_RATE);

			Settings.LU_REGULARIZATION_TERM = Double.valueOf((String) prop.get("lu_regularization_term"));
			System.out.println("lu_regularization_term " + Settings.LU_REGULARIZATION_TERM);

			Settings.FRACTION = Double.valueOf((String) prop.get("fraction"));
			System.out.println("fraction " + Settings.FRACTION);

			Settings.GENRES_NUMBER = Double.valueOf((String) prop.get("genres"));
			System.out.println("genres " + Settings.GENRES_NUMBER);

		} catch (IOException io) {
			io.printStackTrace();
		} finally {
			if (input != null) {
				try {
					input.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}

		}
	}
}

import org.grouplens.lenskit.ItemRecommender;
import org.grouplens.lenskit.core.LenskitConfiguration;
import org.grouplens.lenskit.core.LenskitRecommender;
import org.grouplens.lenskit.data.dao.EventDAO;
import org.grouplens.lenskit.data.dao.SimpleFileRatingDAO;
import org.grouplens.lenskit.scored.ScoredId;
import util.AlgorithmUtil;
import util.ContentAverageDissimilarity;
import util.Settings;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Properties;

public class RecommenderRunner {
	public static void main(String args[]) throws Exception {
		setParameters();
		ContentAverageDissimilarity.create(Settings.DATASET_CONTENT);
		List<ScoredId> pop = getRecs(AlgorithmUtil.getPop());

		List<ScoredId> lcrdu = getRecs(AlgorithmUtil.getAdvancedLC());

		List<ScoredId> svd = getRecs(AlgorithmUtil.getSVD());

		List<ScoredId> pureSVD = getRecs(AlgorithmUtil.getPureSVD());

		List<ScoredId> zheng = getRecs(AlgorithmUtil.getZhengSVDContent());

		List<ScoredId> spr = getRecs(AlgorithmUtil.getLuSVDHinge10000());

		List<ScoredId> pr = getRecs(AlgorithmUtil.getLuSVDBasic());

		System.out.println("Pop");
		System.out.println(pop);
		System.out.println("LCRDU");
		System.out.println(lcrdu);
		System.out.println("SVD");
		System.out.println(svd);
		System.out.println("PureSVD");
		System.out.println(pureSVD);
		System.out.println("Zheng");
		System.out.println(zheng);
		System.out.println("SPR");
		System.out.println(spr);
		System.out.println("PR");
		System.out.println(pr);
	}

	private static List<ScoredId> getRecs(LenskitConfiguration configuration) throws Exception {
		configuration.bind(EventDAO.class).to(new SimpleFileRatingDAO(new File(Settings.DATASET), "\t"));
		LenskitRecommender pop = LenskitRecommender.build(configuration);
		ItemRecommender itemRecommender = pop.getItemRecommender();
		List<ScoredId> recs = itemRecommender.recommend(1000l, 10);
		return recs;
	}

	private static void setParameters() {
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

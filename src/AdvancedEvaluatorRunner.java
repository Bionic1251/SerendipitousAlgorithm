import evaluationMetric.SerendipityTopNMetric;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.baseline.ItemMeanRatingItemScorer;
import org.grouplens.lenskit.baseline.UserMeanBaseline;
import org.grouplens.lenskit.baseline.UserMeanItemScorer;
import org.grouplens.lenskit.core.LenskitConfiguration;
import org.grouplens.lenskit.data.source.DataSource;
import org.grouplens.lenskit.data.source.GenericDataSource;
import org.grouplens.lenskit.data.text.DelimitedColumnEventFormat;
import org.grouplens.lenskit.data.text.RatingEventType;
import org.grouplens.lenskit.data.text.TextEventDAO;
import org.grouplens.lenskit.eval.data.crossfold.CrossfoldTask;
import org.grouplens.lenskit.eval.metrics.predict.RMSEPredictMetric;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors;
import org.grouplens.lenskit.eval.metrics.topn.NDCGTopNMetric;
import org.grouplens.lenskit.eval.metrics.topn.PrecisionRecallTopNMetric;
import org.grouplens.lenskit.eval.traintest.SimpleEvaluator;
import org.grouplens.lenskit.iterative.IterationCount;
import org.grouplens.lenskit.iterative.LearningRate;
import org.grouplens.lenskit.mf.funksvd.*;
import org.hamcrest.Matchers;
import pop.PopItemScorer;
import spr.SPRFunkSVDItemScorer;
import lu.LuFunkSVDItemScorer;

import java.io.File;

public class AdvancedEvaluatorRunner {
	private static final int CROSSFOLD_NUMBER = 1;
	private static final int MY_HOLDOUT_NUMBER = 3;
	private static final int HOLDOUT_NUMBER = 10;
	private static final int MY_AT_N = 5;
	private static final int AT_N = 5;
	private static final int MY_N_MAX = 5;
	private static final int N_MAX = 100;
	private static final int MY_SERENDIPITOUS_ITEMS_NUMBER = 100;
	private static final int SERENDIPITOUS_ITEMS_NUMBER = 100;
	private static final double THRESHOLD = 3.0;
	private static final String MY_DATASET = "D:\\bigdata\\movielens\\fake\\all_ratings_extended";
	private static final String SMALL_DATASET = "D:\\bigdata\\movielens\\ml-100k\\u.data";
	private static final String BIG_DATASET = "D:\\bigdata\\movielens\\ml-1m\\ratings.dat";
	private static final String TRAIN_TEST_FOLDER_NAME = "task";
	private static final String OUTPUT_PATH = "./results/out.csv";
	private static final String OUTPUT_USER_PATH = "./results/user.csv";
	private static final String OUTPUT_ITEM_PATH = "./results/item.csv";

	private static final String MY = "my";
	private static final String SMALL = "small";
	private static final String BIG = "BIG";
	private static final String STATE = SMALL;

	private static void setEvaluator(SimpleEvaluator evaluator) {
		String path = "";
		int holdout = 1;
		DelimitedColumnEventFormat eventFormat = new DelimitedColumnEventFormat(new RatingEventType());
		if (STATE.equals(MY)) {
			holdout = MY_HOLDOUT_NUMBER;
			path = MY_DATASET;
		} else if (STATE.equals(SMALL)) {
			holdout = HOLDOUT_NUMBER;
			path = SMALL_DATASET;
		} else if (STATE.equals(BIG)) {
			eventFormat.setDelimiter("::");
			holdout = HOLDOUT_NUMBER;
			path = BIG_DATASET;
		}
		DataSource dataSource = new GenericDataSource("split", new TextEventDAO(new File(path), eventFormat));
		CrossfoldTask task = new CrossfoldTask(TRAIN_TEST_FOLDER_NAME);
		task.setHoldout(holdout);
		task.setPartitions(CROSSFOLD_NUMBER);
		task.setSource(dataSource);
		evaluator.addDataset(task);

		evaluator.setOutputPath(OUTPUT_PATH);
		evaluator.setUserOutputPath(OUTPUT_USER_PATH);
		evaluator.setPredictOutputPath(OUTPUT_ITEM_PATH);
	}

	public static void main(String args[]) {
		SimpleEvaluator evaluator = new SimpleEvaluator();
		setEvaluator(evaluator);

		LenskitConfiguration LuSVD = new LenskitConfiguration();
		LuSVD.bind(ItemScorer.class).to(LuFunkSVDItemScorer.class);
		LuSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		LuSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		LuSVD.set(FeatureCount.class).to(3);
		LuSVD.set(LearningRate.class).to(0.001);
		LuSVD.set(IterationCount.class).to(2);
		//LuSVD.set(InitialFeatureValue.class).to(2.5);
		evaluator.addAlgorithm("LuSVD", LuSVD);

		LenskitConfiguration SPRSVD = new LenskitConfiguration();
		SPRSVD.bind(ItemScorer.class).to(SPRFunkSVDItemScorer.class);
		SPRSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		SPRSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		SPRSVD.set(FeatureCount.class).to(3);
		SPRSVD.set(LearningRate.class).to(0.001);
		SPRSVD.set(IterationCount.class).to(5);
		//SPRSVD.set(InitialFeatureValue.class).to(2.5);
		evaluator.addAlgorithm("SPRSVD", SPRSVD);

		LenskitConfiguration POP = new LenskitConfiguration();
		POP.bind(ItemScorer.class).to(PopItemScorer.class);
		POP.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		POP.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		evaluator.addAlgorithm("POP", POP);

		LenskitConfiguration SVD = new LenskitConfiguration();
		SVD.bind(ItemScorer.class).to(FunkSVDItemScorer.class);
		SVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		SVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		SVD.set(FeatureCount.class).to(3);
		SVD.set(IterationCount.class).to(500);
		//SVD.set(LearningRate.class).to(0.01);
		evaluator.addAlgorithm("SVD", SVD);

		addMetrics(evaluator);

		try {
			evaluator.call();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void addMetrics(SimpleEvaluator evaluator) {
		int at_n, serendipitousNumber;
		if (STATE.equals(MY)) {
			serendipitousNumber = MY_SERENDIPITOUS_ITEMS_NUMBER;
			at_n = MY_AT_N;
		} else {
			serendipitousNumber = SERENDIPITOUS_ITEMS_NUMBER;
			at_n = AT_N;
		}
		String suffix = at_n + "";
		ItemSelector threshold = ItemSelectors.testRatingMatches(Matchers.greaterThan(THRESHOLD));
		ItemSelector candidates = ItemSelectors.allItems();
		ItemSelector exclude = ItemSelectors.trainingItems();
		evaluator.addMetric(new RMSEPredictMetric());
		evaluator.addMetric(new PrecisionRecallTopNMetric("", suffix, at_n, candidates, exclude, threshold));
		evaluator.addMetric(new NDCGTopNMetric("", suffix, at_n, candidates, exclude));
		evaluator.addMetric(new SerendipityTopNMetric(suffix, at_n, serendipitousNumber, candidates, exclude, threshold));
	}
}

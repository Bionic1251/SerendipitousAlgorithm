import annotation.Alpha;
import annotation.Threshold;
import evaluationMetric.SerendipityTopNMetric;
import funkSVD.lu.LuFunkSVDItemScorerBaysian;
import funkSVD.zheng.ZhengFunkSVDItemScorer;
import it.unimi.dsi.fastutil.longs.LongSet;
import mf.baseline.SVDItemScorer;
import mf.lu.LuSVDItemScorer;
import mf.zheng.ZhengSVDItemScorer;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.baseline.ItemMeanRatingItemScorer;
import org.grouplens.lenskit.baseline.UserMeanBaseline;
import org.grouplens.lenskit.baseline.UserMeanItemScorer;
import org.grouplens.lenskit.core.LenskitConfiguration;
import org.grouplens.lenskit.cursors.Cursor;
import org.grouplens.lenskit.data.dao.ItemEventDAO;
import org.grouplens.lenskit.data.event.Event;
import org.grouplens.lenskit.data.history.ItemEventCollection;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.data.source.DataSource;
import org.grouplens.lenskit.data.source.GenericDataSource;
import org.grouplens.lenskit.data.text.DelimitedColumnEventFormat;
import org.grouplens.lenskit.data.text.RatingEventType;
import org.grouplens.lenskit.data.text.TextEventDAO;
import org.grouplens.lenskit.eval.data.crossfold.CrossfoldTask;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors;
import org.grouplens.lenskit.eval.metrics.topn.NDCGTopNMetric;
import org.grouplens.lenskit.eval.metrics.topn.PrecisionRecallTopNMetric;
import org.grouplens.lenskit.eval.traintest.SimpleEvaluator;
import org.grouplens.lenskit.iterative.IterationCount;
import org.grouplens.lenskit.iterative.LearningRate;
import org.grouplens.lenskit.knn.NeighborhoodSize;
import org.grouplens.lenskit.knn.item.ItemItemScorer;
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.FunkSVDItemScorer;
import org.grouplens.lenskit.util.ScoredItemAccumulator;
import org.grouplens.lenskit.util.TopNScoredItemAccumulator;
import org.grouplens.lenskit.vectors.similarity.PearsonCorrelation;
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity;
import org.hamcrest.Matchers;
import pop.PopItemScorer;
import random.RandomItemScorer;

import java.io.File;

public class ExperimentRunner {
	private static final int CROSSFOLD_NUMBER = 1;
	private static final int HOLDOUT_NUMBER = 10;
	private static final int AT_N = 5;
	private static final int N_MAX = 5;
	private static final int EXPECTED_ITEMS_NUMBER = 100;
	private static final double THRESHOLD = 3.0;
	private static final String DATASET = "D:\\bigdata\\movielens\\hetrec\\user_ratedmovies-timestamps.dat";
	private static final String TRAIN_TEST_FOLDER_NAME = "release";
	private static final String OUTPUT_PATH = "./releaseResults/out.csv";
	private static final String OUTPUT_USER_PATH = "./releaseResults/user.csv";
	private static final String OUTPUT_ITEM_PATH = "./releaseResults/item.csv";

	private static final double MIN = 0;
	private static final double MAX = 5;
	private static final int POPULAR_ITEMS_NUMBER = 50;
	private static final int FEATURE_COUNT = 5;
	private static final int ALL_ITEMS_ITERATION_COUNT = 5;
	private static final int ITERATION_COUNT = 5000;
	private static final double ALPHA = 0.5;

	private static void setEvaluator(SimpleEvaluator evaluator) {
		DelimitedColumnEventFormat eventFormat = new DelimitedColumnEventFormat(new RatingEventType());
		DataSource dataSource = new GenericDataSource("split", new TextEventDAO(new File(DATASET), eventFormat), new PreferenceDomain(MIN, MAX));
		CrossfoldTask task = new CrossfoldTask(TRAIN_TEST_FOLDER_NAME);
		task.setHoldout(HOLDOUT_NUMBER);
		task.setPartitions(CROSSFOLD_NUMBER);
		task.setSource(dataSource);
		evaluator.addDataset(task);

		evaluator.setOutputPath(OUTPUT_PATH);
		evaluator.setUserOutputPath(OUTPUT_USER_PATH);
		evaluator.setPredictOutputPath(OUTPUT_ITEM_PATH);
	}

	private static LenskitConfiguration getRandom() {
		LenskitConfiguration random = new LenskitConfiguration();
		random.bind(ItemScorer.class).to(RandomItemScorer.class);
		return random;
	}

	private static LenskitConfiguration getZhengFunkSVD() {
		LenskitConfiguration ZhengFunkSVD = new LenskitConfiguration();
		ZhengFunkSVD.bind(ItemScorer.class).to(ZhengFunkSVDItemScorer.class);
		ZhengFunkSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		ZhengFunkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		ZhengFunkSVD.set(FeatureCount.class).to(FEATURE_COUNT);
		ZhengFunkSVD.set(IterationCount.class).to(ALL_ITEMS_ITERATION_COUNT);
		ZhengFunkSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		ZhengFunkSVD.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		return ZhengFunkSVD;
	}

	public static void main(String args[]) {
		SimpleEvaluator evaluator = new SimpleEvaluator();
		setEvaluator(evaluator);

		evaluator.addAlgorithm("Random", getRandom());

		evaluator.addAlgorithm("ZhengFunkSVD", getZhengFunkSVD());

		LenskitConfiguration LuFunkSVD = new LenskitConfiguration();
		LuFunkSVD.bind(ItemScorer.class).to(LuFunkSVDItemScorerBaysian.class);
		LuFunkSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		LuFunkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		LuFunkSVD.set(FeatureCount.class).to(FEATURE_COUNT);
		LuFunkSVD.set(LearningRate.class).to(0.00001);
		LuFunkSVD.set(IterationCount.class).to(ALL_ITEMS_ITERATION_COUNT);
		LuFunkSVD.set(Threshold.class).to(THRESHOLD);
		LuFunkSVD.set(Alpha.class).to(ALPHA);
		evaluator.addAlgorithm("LuFunkSVD", LuFunkSVD);

		LenskitConfiguration FunkSVD = new LenskitConfiguration();
		FunkSVD.bind(ItemScorer.class).to(FunkSVDItemScorer.class);
		FunkSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		FunkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		FunkSVD.set(FeatureCount.class).to(FEATURE_COUNT);
		FunkSVD.set(IterationCount.class).to(ITERATION_COUNT);
		evaluator.addAlgorithm("funkSVD", FunkSVD);

		LenskitConfiguration LuSVD = new LenskitConfiguration();
		LuSVD.bind(ItemScorer.class).to(LuSVDItemScorer.class);
		LuSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		LuSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		LuSVD.set(FeatureCount.class).to(FEATURE_COUNT);
		LuSVD.set(LearningRate.class).to(0.001);
		LuSVD.set(IterationCount.class).to(ALL_ITEMS_ITERATION_COUNT);
		LuSVD.set(Threshold.class).to(THRESHOLD);
		LuSVD.set(Alpha.class).to(ALPHA);
		evaluator.addAlgorithm("LuSVD", LuSVD);

		LenskitConfiguration SVDBaseline = new LenskitConfiguration();
		SVDBaseline.bind(ItemScorer.class).to(SVDItemScorer.class);
		SVDBaseline.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		SVDBaseline.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		SVDBaseline.set(FeatureCount.class).to(FEATURE_COUNT);
		SVDBaseline.set(IterationCount.class).to(ITERATION_COUNT);
		evaluator.addAlgorithm("baseline", SVDBaseline);

		LenskitConfiguration ZhengSVD = new LenskitConfiguration();
		ZhengSVD.bind(ItemScorer.class).to(ZhengSVDItemScorer.class);
		ZhengSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		ZhengSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		ZhengSVD.set(FeatureCount.class).to(FEATURE_COUNT);
		ZhengSVD.set(IterationCount.class).to(ITERATION_COUNT);
		ZhengSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		ZhengSVD.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		evaluator.addAlgorithm("ZhengSVD", ZhengSVD);

		LenskitConfiguration POP = new LenskitConfiguration();
		POP.bind(ItemScorer.class).to(PopItemScorer.class);
		evaluator.addAlgorithm("POP", POP);

		LenskitConfiguration itemItem = new LenskitConfiguration();
		itemItem.bind(ItemScorer.class).to(ItemItemScorer.class);
		itemItem.bind(BaselineScorer.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		itemItem.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		evaluator.addAlgorithm("itemItem", itemItem);

		addEvaluationMetrics(evaluator);

		try {
			evaluator.call();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void addEvaluationMetrics(SimpleEvaluator evaluator) {
		ItemSelector popCandidates = ItemSelectors.union(new MyPopularItemSelector(getPopItems()), ItemSelectors.testItems());
		addMetricsWithParameters(evaluator, N_MAX, popCandidates, POPULAR_ITEMS_NUMBER + "candidates");

		addMetricsWithParameters(evaluator, N_MAX, ItemSelectors.allItems(), "all");

		addMetricsWithParameters(evaluator, N_MAX, ItemSelectors.testItems(), "test");

		//addMetricsWithParameters(evaluator, N_MAX, ItemSelectors.union(ItemSelectors.testItems(), ItemSelectors.nRandom(POPULAR_ITEMS_NUMBER)), POPULAR_ITEMS_NUMBER + "random");
	}

	private static void addMetricsWithParameters(SimpleEvaluator evaluator, int maxNumber, ItemSelector candidates, String prefix) {
		ItemSelector threshold = ItemSelectors.testRatingMatches(Matchers.greaterThan(THRESHOLD));
		ItemSelector exclude = ItemSelectors.trainingItems();
		for (int i = AT_N; i <= maxNumber; i += 5) {
			String suffix = prefix + "." + i;
			evaluator.addMetric(new PrecisionRecallTopNMetric("", suffix, i, candidates, exclude, threshold));
			evaluator.addMetric(new NDCGTopNMetric("", suffix, i, candidates, exclude));
			evaluator.addMetric(new SerendipityTopNMetric(suffix, i, EXPECTED_ITEMS_NUMBER, candidates, exclude, threshold));
		}
	}

	private static LongSet getPopItems() {
		DelimitedColumnEventFormat eventFormat = new DelimitedColumnEventFormat(new RatingEventType());
		DataSource dataSource = new GenericDataSource("split", new TextEventDAO(new File(DATASET), eventFormat), new PreferenceDomain(MIN, MAX));
		ItemEventDAO idao = dataSource.getItemEventDAO();
		ScoredItemAccumulator accum = new TopNScoredItemAccumulator(POPULAR_ITEMS_NUMBER);
		Cursor<ItemEventCollection<Event>> items = idao.streamEventsByItem();
		try {
			for (ItemEventCollection<Event> item : items) {
				accum.put(item.getItemId(), item.size());
			}
		} finally {
			items.close();
		}
		return accum.finishSet();
	}
}

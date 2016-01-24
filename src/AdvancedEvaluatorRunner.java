import adamopoulos.AdaItemScorer;
import annotation.RatingPredictor;
import util.AlgorithmUtil;
import util.ContentUtil;
import evaluationMetric.PopSerendipityTopNMetric;
import evaluationMetric.SerendipityTopNMetric;
import funkSVD.zheng.ZhengFunkSVDItemScorer;
import it.unimi.dsi.fastutil.longs.LongSet;
import mf.baseline.SVDItemScorer;
import annotation.Alpha;
import annotation.Threshold;
import mf.lu.LuSVDItemScorer;
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
import org.grouplens.lenskit.mf.funksvd.*;
import org.grouplens.lenskit.util.ScoredItemAccumulator;
import org.grouplens.lenskit.util.TopNScoredItemAccumulator;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.similarity.PearsonCorrelation;
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity;
import org.hamcrest.Matchers;
import pop.PopItemScorer;
import mf.zheng.ZhengSVDItemScorer;
import random.RandomItemScorer;
import util.MyPopularItemSelector;

import java.io.File;
import java.util.*;

public class AdvancedEvaluatorRunner {
	private static final int CROSSFOLD_NUMBER = 1;
	private static final int MY_HOLDOUT_NUMBER = 3;
	private static final int HOLDOUT_NUMBER = 10;
	private static final int MY_AT_N = 5;
	private static final int AT_N = 20;
	private static final int MY_EXPECTED_ITEMS_NUMBER = 2;
	private static final int EXPECTED_ITEMS_NUMBER = 40;
	private static final double THRESHOLD = 3.0;
	private static final int POPULAR_ITEMS_NUMBER = 50;
	private static final String MY_DATASET = "D:\\bigdata\\movielens\\fake\\all_ratings_extended";
	private static final String SMALL_DATASET = "D:\\bigdata\\movielens\\ml-100k\\u.data";
	private static final String SMALL_DATASET_CONTENT = "D:\\bigdata\\movielens\\ml-100k\\small_content.dat";
	private static final String BIG_DATASET = "D:\\bigdata\\movielens\\hetrec\\user_ratedmovies-timestamps.dat";
	private static final String BIG_DATASET_CONTENT = "D:\\bigdata\\movielens\\hetrec\\big_content.dat";
	private static final String TRAIN_TEST_FOLDER_NAME = "task";
	private static final String OUTPUT_PATH = "./results/out.csv";
	private static final String OUTPUT_USER_PATH = "./results/user.csv";
	private static final String OUTPUT_ITEM_PATH = "./results/item.csv";

	private static final double MIN = 0;
	private static final double MAX = 5;

	private static final String MY = "my";
	private static final String SMALL = "small";
	private static final String BIG = "big";
	private static final String STATE = SMALL;

	private static String path;
	private static String contentPath;
	private static DelimitedColumnEventFormat eventFormat;
	private static Map<Long, SparseVector> itemContentMap;

	private static final int FEATURE_COUNT = 3;

	private static void setEvaluator(SimpleEvaluator evaluator) {
		int holdout = MY_HOLDOUT_NUMBER;
		eventFormat = new DelimitedColumnEventFormat(new RatingEventType());
		if (STATE.equals(MY)) {
			holdout = MY_HOLDOUT_NUMBER;
			path = MY_DATASET;
		} else if (STATE.equals(SMALL)) {
			holdout = HOLDOUT_NUMBER;
			path = SMALL_DATASET;
			contentPath = SMALL_DATASET_CONTENT;
		} else if (STATE.equals(BIG)) {
			//eventFormat.setDelimiter("::");
			holdout = HOLDOUT_NUMBER;
			path = BIG_DATASET;
			contentPath = BIG_DATASET_CONTENT;
		}
		DataSource dataSource = new GenericDataSource("split", new TextEventDAO(new File(path), eventFormat), new PreferenceDomain(MIN, MAX));
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

		itemContentMap = ContentUtil.getItemContentMap(contentPath);

		evaluator.addAlgorithm("POP", AlgorithmUtil.getPop());
		//evaluator.addAlgorithm("LuFunkSVDBaysian", AlgorithmUtil.getLuFunkSVDBaysian(FEATURE_COUNT));
		//evaluator.addAlgorithm("LuFunkSVDHinge", AlgorithmUtil.getLuFunkSVDHinge(FEATURE_COUNT));
		//evaluator.addAlgorithm("FunkSVD", AlgorithmUtil.getFunkSVD(FEATURE_COUNT));
		//evaluator.addAlgorithm("LuSVDHinge", AlgorithmUtil.getLuSVDHinge(FEATURE_COUNT));
		//evaluator.addAlgorithm("LuSVDBaysian", AlgorithmUtil.getLuSVDBaysian(FEATURE_COUNT));
		//evaluator.addAlgorithm("SVD", AlgorithmUtil.getSVD(FEATURE_COUNT));

		LenskitConfiguration rnd = new LenskitConfiguration();
		rnd.bind(ItemScorer.class).to(RandomItemScorer.class);
		//evaluator.addAlgorithm("Random", rnd);

		LenskitConfiguration adaSVD = new LenskitConfiguration();
		adaSVD.bind(ItemScorer.class).to(AdaItemScorer.class);
		adaSVD.bind(RatingPredictor.class, ItemScorer.class).to(SVDItemScorer.class);
		adaSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		adaSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		adaSVD.set(FeatureCount.class).to(2);
		adaSVD.set(IterationCount.class).to(3000);
		adaSVD.set(Threshold.class).to(THRESHOLD);
		adaSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		//evaluator.addAlgorithm("AdaSVD", adaSVD);

		LenskitConfiguration adaFunkSVD = new LenskitConfiguration();
		adaFunkSVD.bind(ItemScorer.class).to(AdaItemScorer.class);
		adaFunkSVD.bind(RatingPredictor.class, ItemScorer.class).to(FunkSVDItemScorer.class);
		adaFunkSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		adaFunkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		adaFunkSVD.set(FeatureCount.class).to(5);
		adaFunkSVD.set(IterationCount.class).to(3000);
		adaFunkSVD.set(Threshold.class).to(THRESHOLD);
		adaFunkSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		//evaluator.addAlgorithm("AdaFunkSVD", adaFunkSVD);

		LenskitConfiguration ZhengFunkSVD = new LenskitConfiguration();
		ZhengFunkSVD.bind(ItemScorer.class).to(ZhengFunkSVDItemScorer.class);
		ZhengFunkSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		ZhengFunkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		ZhengFunkSVD.set(FeatureCount.class).to(5);
		ZhengFunkSVD.set(IterationCount.class).to(3);
		ZhengFunkSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		ZhengFunkSVD.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		//evaluator.addAlgorithm("ZhengFunkSVD", ZhengFunkSVD);

		LenskitConfiguration SVDBaseline = new LenskitConfiguration();
		SVDBaseline.bind(ItemScorer.class).to(SVDItemScorer.class);
		SVDBaseline.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		SVDBaseline.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		SVDBaseline.set(FeatureCount.class).to(5);
		SVDBaseline.set(IterationCount.class).to(3000);
		//evaluator.addAlgorithm("SVDBaseline", SVDBaseline);

		LenskitConfiguration ZhengSVD = new LenskitConfiguration();
		ZhengSVD.bind(ItemScorer.class).to(ZhengSVDItemScorer.class);
		ZhengSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		ZhengSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		ZhengSVD.set(FeatureCount.class).to(5);
		ZhengSVD.set(IterationCount.class).to(500);
		ZhengSVD.set(NeighborhoodSize.class).to(Integer.MAX_VALUE);
		ZhengSVD.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		//evaluator.addAlgorithm("ZhengSVD", ZhengSVD);

		LenskitConfiguration itemItem = new LenskitConfiguration();
		itemItem.bind(ItemScorer.class).to(ItemItemScorer.class);
		itemItem.bind(BaselineScorer.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		itemItem.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
		//evaluator.addAlgorithm("itemItem", itemItem);

		addEvaluationMetrics(evaluator);

		try {
			evaluator.call();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void addEvaluationMetrics(SimpleEvaluator evaluator) {
		int at_n, serendipitousNumber;
		if (STATE.equals(MY)) {
			serendipitousNumber = MY_EXPECTED_ITEMS_NUMBER;
			at_n = MY_AT_N;
		} else {
			serendipitousNumber = EXPECTED_ITEMS_NUMBER;
			at_n = AT_N;
		}

		ItemSelector popCandidates = ItemSelectors.union(new MyPopularItemSelector(getPopItems()), ItemSelectors.testItems());
		addMetricsWithParameters(evaluator, at_n, popCandidates, POPULAR_ITEMS_NUMBER + "pop");

		addMetricsWithParameters(evaluator, at_n, ItemSelectors.allItems(), "all");

		addMetricsWithParameters(evaluator, 5, ItemSelectors.testItems(), "test");

		addMetricsWithParameters(evaluator, at_n, ItemSelectors.union(ItemSelectors.testItems(), ItemSelectors.nRandom(POPULAR_ITEMS_NUMBER)), POPULAR_ITEMS_NUMBER + "rand");
	}

	private static void addMetricsWithParameters(SimpleEvaluator evaluator, int maxNumber, ItemSelector candidates, String prefix) {
		ItemSelector threshold = ItemSelectors.testRatingMatches(Matchers.greaterThan(THRESHOLD));
		ItemSelector exclude = ItemSelectors.trainingItems();
		for (int i = 5; i <= maxNumber; i += 5) {
			String suffix = prefix + "." + i;
			evaluator.addMetric(new PrecisionRecallTopNMetric("", suffix, i, candidates, exclude, threshold));
			evaluator.addMetric(new NDCGTopNMetric("", suffix, i, candidates, exclude));
			evaluator.addMetric(new PopSerendipityTopNMetric(suffix, i, EXPECTED_ITEMS_NUMBER, candidates, exclude, threshold));
			evaluator.addMetric(new SerendipityTopNMetric("content." + suffix, i, EXPECTED_ITEMS_NUMBER, candidates, exclude, threshold, itemContentMap, 300));
		}
	}

	/*private static void addMetrics(SimpleEvaluator evaluator) {
		int at_n, serendipitousNumber;
		if (STATE.equals(MY)) {
			serendipitousNumber = MY_EXPECTED_ITEMS_NUMBER;
			at_n = MY_AT_N;
		} else {
			serendipitousNumber = EXPECTED_ITEMS_NUMBER;
			at_n = AT_N;
		}
		ItemSelector threshold = ItemSelectors.testRatingMatches(Matchers.greaterThan(THRESHOLD));
		ItemSelector candidates = ItemSelectors.union(new MyPopularItemSelector(getPopItems()), ItemSelectors.testItems());
		ItemSelector exclude = ItemSelectors.trainingItems();

		String suffix = at_n + "";
		//evaluator.addMetric(new RMSEPredictMetric());
		evaluator.addMetric(new PrecisionRecallTopNMetric("", suffix, at_n, candidates, exclude, threshold));
		evaluator.addMetric(new NDCGTopNMetric("", suffix, at_n, candidates, exclude));
		evaluator.addMetric(new PopSerendipityTopNMetric(suffix, at_n, serendipitousNumber, candidates, exclude, threshold));
		evaluator.addMetric(new SerendipityTopNMetric("cor" + suffix, at_n, serendipitousNumber, candidates, exclude, threshold, itemContentMap, 500));

		*//*for (int i = at_n; i <= 20; i += 5) {
			String suffix = i + "";
			//evaluator.addMetric(new RMSEPredictMetric());
			evaluator.addMetric(new PrecisionRecallTopNMetric("", suffix, i, candidates, exclude, threshold));
			evaluator.addMetric(new NDCGTopNMetric("", suffix, i, candidates, exclude));
			evaluator.addMetric(new SerendipityTopNMetric(suffix, i, serendipitousNumber, candidates, exclude, threshold));
		}*//*
	}*/

	private static LongSet getPopItems() {
		DataSource dataSource = new GenericDataSource("split", new TextEventDAO(new File(path), eventFormat), new PreferenceDomain(MIN, MAX));
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

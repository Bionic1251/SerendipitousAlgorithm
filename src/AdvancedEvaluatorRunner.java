import adamopoulos.AdaItemScorer;
import annotation.RatingPredictor;
import evaluationMetric.PopSerendipityTopNMetric;
import evaluationMetric.SerendipityTopNMetric;
import funkSVD.lu.LuFunkSVDItemScorerBaysian;
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
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.similarity.PearsonCorrelation;
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity;
import org.hamcrest.Matchers;
import pop.PopItemScorer;
import mf.zheng.ZhengSVDItemScorer;
import random.RandomItemScorer;

import java.io.BufferedReader;
import java.io.File;
import java.util.*;

public class AdvancedEvaluatorRunner {
	private static final int CROSSFOLD_NUMBER = 1;
	private static final int MY_HOLDOUT_NUMBER = 3;
	private static final int HOLDOUT_NUMBER = 10;
	private static final int MY_AT_N = 5;
	private static final int AT_N = 10;
	private static final int MY_SERENDIPITOUS_ITEMS_NUMBER = 2;
	private static final int SERENDIPITOUS_ITEMS_NUMBER = 40;
	private static final double THRESHOLD = 3.0;
	private static final String MY_DATASET = "D:\\bigdata\\movielens\\fake\\all_ratings_extended";
	private static final String SMALL_DATASET = "D:\\bigdata\\movielens\\ml-100k\\u.data";
	private static final String SMALL_DATASET_CONTENT = "D:\\bigdata\\movielens\\ml-100k\\u.item";
	private static final String BIG_DATASET = "D:\\bigdata\\movielens\\hetrec\\user_ratedmovies-timestamps.dat";
	private static final String BIG_DATASET_CONTENT = "D:\\bigdata\\movielens\\ml-100k\\u.item";
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

		prepare();

		LenskitConfiguration POP = new LenskitConfiguration();
		POP.bind(ItemScorer.class).to(PopItemScorer.class);
		evaluator.addAlgorithm("POP", POP);

		LenskitConfiguration rnd = new LenskitConfiguration();
		rnd.bind(ItemScorer.class).to(RandomItemScorer.class);
		//evaluator.addAlgorithm("Random", rnd);

		LenskitConfiguration adaSVD = new LenskitConfiguration();
		adaSVD.bind(ItemScorer.class).to(AdaItemScorer.class);
		adaSVD.bind(RatingPredictor.class, ItemScorer.class).to(SVDItemScorer.class);
		adaSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		adaSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		adaSVD.set(FeatureCount.class).to(5);
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

		LenskitConfiguration LuFunkSVD = new LenskitConfiguration();
		LuFunkSVD.bind(ItemScorer.class).to(LuFunkSVDItemScorerBaysian.class);
		LuFunkSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		LuFunkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		LuFunkSVD.set(FeatureCount.class).to(5);
		LuFunkSVD.set(LearningRate.class).to(0.00001);
		LuFunkSVD.set(IterationCount.class).to(20);
		LuFunkSVD.set(Threshold.class).to(THRESHOLD);
		LuFunkSVD.set(Alpha.class).to(0.5);
		//evaluator.addAlgorithm("LuFunkSVD", LuFunkSVD);

		LenskitConfiguration FunkSVD = new LenskitConfiguration();
		FunkSVD.bind(ItemScorer.class).to(FunkSVDItemScorer.class);
		FunkSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		FunkSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		FunkSVD.set(FeatureCount.class).to(5);
		FunkSVD.set(IterationCount.class).to(3000);
		//evaluator.addAlgorithm("funkSVD", FunkSVD);

		LenskitConfiguration LuSVD = new LenskitConfiguration();
		LuSVD.bind(ItemScorer.class).to(LuSVDItemScorer.class);
		LuSVD.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
		LuSVD.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
		LuSVD.set(FeatureCount.class).to(5);
		LuSVD.set(LearningRate.class).to(0.001);
		LuSVD.set(IterationCount.class).to(10);
		LuSVD.set(Threshold.class).to(THRESHOLD);
		LuSVD.set(Alpha.class).to(0.5);
		//evaluator.addAlgorithm("LuSVD", LuSVD);

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
		ItemSelector threshold = ItemSelectors.testRatingMatches(Matchers.greaterThan(THRESHOLD));
		ItemSelector candidates = ItemSelectors.union(new MyPopularItemSelector(getPopItems()), ItemSelectors.testItems());
		ItemSelector exclude = ItemSelectors.trainingItems();

		String suffix = at_n + "";
		//evaluator.addMetric(new RMSEPredictMetric());
		evaluator.addMetric(new PrecisionRecallTopNMetric("", suffix, at_n, candidates, exclude, threshold));
		evaluator.addMetric(new NDCGTopNMetric("", suffix, at_n, candidates, exclude));
		evaluator.addMetric(new PopSerendipityTopNMetric(suffix, at_n, serendipitousNumber, candidates, exclude, threshold));
		evaluator.addMetric(new SerendipityTopNMetric("cor" + suffix, at_n, serendipitousNumber, candidates, exclude, threshold, itemContentMap, 500));

		/*for (int i = at_n; i <= 20; i += 5) {
			String suffix = i + "";
			//evaluator.addMetric(new RMSEPredictMetric());
			evaluator.addMetric(new PrecisionRecallTopNMetric("", suffix, i, candidates, exclude, threshold));
			evaluator.addMetric(new NDCGTopNMetric("", suffix, i, candidates, exclude));
			evaluator.addMetric(new SerendipityTopNMetric(suffix, i, serendipitousNumber, candidates, exclude, threshold));
		}*/
	}

	private static LongSet getPopItems() {
		DataSource dataSource = new GenericDataSource("split", new TextEventDAO(new File(path), eventFormat), new PreferenceDomain(MIN, MAX));
		ItemEventDAO idao = dataSource.getItemEventDAO();
		ScoredItemAccumulator accum = new TopNScoredItemAccumulator(1000);
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

	private static int[] termDocFreq;
	private static Map<Long, SparseVector> itemContentMap;

	private static void prepare() {
		itemContentMap = new HashMap<Long, SparseVector>();
		Map<Long, BitSet> vecMap = getVectors();
		for (Map.Entry<Long, BitSet> entry : vecMap.entrySet()) {
			SparseVector vector = vecByBitSet(entry.getValue(), vecMap.size());
			itemContentMap.put(entry.getKey(), vector);
		}
	}

	private static SparseVector vecByBitSet(BitSet set, int docNum) {
		Set<Long> keys = new HashSet<Long>();
		for (int i = 0; i < termDocFreq.length; i++) {
			keys.add((long) i);
		}
		MutableSparseVector vector = MutableSparseVector.create(keys);
		for (int i = 0; i < termDocFreq.length; i++) {
			if (set.get(i)) {
				double tfidf = getTFIDF(docNum, i);
				vector.set((long) i, tfidf);
			}
		}
		return vector;
	}

	private static double getTFIDF(int docNum, int termNumber) {
		double tf = 1.0 / termDocFreq.length;
		double idf = Math.log((double) docNum / termDocFreq[termNumber]);
		return tf * idf;
	}

	private static Map<Long, BitSet> getVectors() {
		Map<Long, BitSet> vecMap = new HashMap<Long, BitSet>();
		int boolLen = 23, boolStart = 5;
		termDocFreq = new int[boolLen - boolStart + 1];
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(contentPath));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] vector = line.split("\\|");
					Long id = Long.valueOf(vector[0]);
					BitSet bitSet = new BitSet(boolLen - boolStart + 1);
					for (int i = boolStart; i <= boolLen; i++) {
						if (Integer.valueOf(vector[i]) != 0) {
							bitSet.set(i - boolStart);
							termDocFreq[i - boolStart]++;
						}
					}
					vecMap.put(id, bitSet);
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return vecMap;
	}
}

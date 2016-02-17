import evaluationMetric.*;
import it.unimi.dsi.fastutil.longs.LongSet;
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
import org.grouplens.lenskit.eval.metrics.predict.CoveragePredictMetric;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors;
import org.grouplens.lenskit.eval.metrics.topn.NDCGTopNMetric;
import org.grouplens.lenskit.eval.metrics.topn.PrecisionRecallTopNMetric;
import org.grouplens.lenskit.eval.traintest.SimpleEvaluator;
import org.grouplens.lenskit.util.ScoredItemAccumulator;
import org.grouplens.lenskit.util.TopNScoredItemAccumulator;
import org.grouplens.lenskit.vectors.SparseVector;
import org.hamcrest.Matchers;
import util.AlgorithmUtil;
import util.ContentUtil;
import util.MyPopularItemSelector;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Map;
import java.util.Properties;

public class ExperimentRunner {
	//properties are overriden in config.properties
	private static int CROSSFOLD_NUMBER = 1;
	private static int HOLDOUT_NUMBER = 15;
	private static int SHORT_HEAD_END = 9;
	private static int LONG_TAIL_START = 79;
	private static int POPULAR_ITEMS_SERENDIPITY_NUMBER = 22;
	private static int RANDOM_ITEMS_FOR_CANDIDATES = 30;
	private static int POPULAR_ITEMS_FOR_CANDIDATES = 200;
	private static String DATASET = "ml/small/ratings.dat";
	private static String DATASET_CONTENT = "ml/small/content.dat";
	private static String TRAIN_TEST_FOLDER_NAME = "task";
	private static String OUTPUT_PATH = "/out.csv";
	private static String OUTPUT_USER_PATH = "/user.csv";
	private static String OUTPUT_ITEM_PATH = "/item.csv";

	private static double D_THRESHOLD = 0.2;
	private static double U_THRESHOLD = 0.2;

	private static double MIN = 0;
	private static double MAX = 5;

	private static DelimitedColumnEventFormat eventFormat;
	public static Map<Long, SparseVector> itemContentMap; //public field used by ZhengSVD. Fix it later

	private static void setEvaluator(SimpleEvaluator evaluator) {
		setParameters();
		eventFormat = new DelimitedColumnEventFormat(new RatingEventType());
		DataSource dataSource = new GenericDataSource("split", new TextEventDAO(new File(DATASET), eventFormat), new PreferenceDomain(MIN, MAX));
		CrossfoldTask task = new CrossfoldTask(TRAIN_TEST_FOLDER_NAME);
		task.setHoldout(HOLDOUT_NUMBER);
		task.setPartitions(CROSSFOLD_NUMBER);
		task.setSource(dataSource);
		evaluator.addDataset(task);

		Date cur = new Date();
		SimpleDateFormat format = new SimpleDateFormat("dd.MM.yy_HH.mm.ss");

		evaluator.setOutputPath("out/" + format.format(cur) + OUTPUT_PATH);
		evaluator.setUserOutputPath("out/" + format.format(cur) + OUTPUT_USER_PATH);
		evaluator.setPredictOutputPath("out/" + format.format(cur) + OUTPUT_ITEM_PATH);
	}

	public static void main(String algs[]) {
		SimpleEvaluator evaluator = new SimpleEvaluator();
		setEvaluator(evaluator);

		addAlgorithms(algs, evaluator);

		itemContentMap = ContentUtil.getItemContentMap(DATASET_CONTENT);
		AlgorithmUtil.itemContentMap = itemContentMap;

		addEvaluationMetrics(evaluator);

		try {
			evaluator.call();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void addAlgorithms(String algs[], SimpleEvaluator evaluator) {
		Map<String, LenskitConfiguration> configurationMap = AlgorithmUtil.getMap();
		if (algs[0].equals("all")) {
			System.out.println("all");
			for (Map.Entry<String, LenskitConfiguration> entry : configurationMap.entrySet()) {
				evaluator.addAlgorithm(entry.getKey(), entry.getValue());
			}
		} else {
			System.out.println("Algorithms");
			for (String alg : algs) {
				System.out.println(alg);
				if (!configurationMap.containsKey(alg)) {
					System.out.println(alg + " doesn't exist");
					System.exit(1);
				}
				evaluator.addAlgorithm(alg, configurationMap.get(alg));
			}
		}
	}

	private static void addEvaluationMetrics(SimpleEvaluator evaluator) {
		addMetricsWithParameters(evaluator, ItemSelectors.testItems(), "test");

		addMetricsWithParameters(evaluator, ItemSelectors.allItems(), "all");

		ItemSelector popCandidates = ItemSelectors.union(new MyPopularItemSelector(getPopItems(POPULAR_ITEMS_FOR_CANDIDATES)), ItemSelectors.testItems());
		//addMetricsWithParameters(evaluator, popCandidates, POPULAR_ITEMS_FOR_CANDIDATES + "pop");

		//addMetricsWithParameters(evaluator, ItemSelectors.union(ItemSelectors.testItems(), ItemSelectors.nRandom(RANDOM_ITEMS_FOR_CANDIDATES)), RANDOM_ITEMS_FOR_CANDIDATES + "rand");
	}

	private static void addMetricsWithParameters(SimpleEvaluator evaluator, ItemSelector candidates, String prefix) {
		ItemSelector threshold = ItemSelectors.testRatingMatches(Matchers.greaterThan(AlgorithmUtil.THRESHOLD));
		ItemSelector exclude = ItemSelectors.trainingItems();
		evaluator.addMetric(new AggregatePrecisionRecallTopNMetric(prefix, "", candidates, exclude, threshold));
		evaluator.addMetric(new AggregateNDCGTopNMetric(prefix, "", candidates, exclude));
		evaluator.addMetric(new AggregatePopSerendipityTopNMetric(prefix, POPULAR_ITEMS_SERENDIPITY_NUMBER, candidates, exclude, threshold));
		evaluator.addMetric(new AggregateSerendipityNDCGMetric("RANK22" + prefix, "", candidates, exclude, AlgorithmUtil.THRESHOLD, itemContentMap, U_THRESHOLD, D_THRESHOLD));
	}

	private static LongSet getPopItems(int popNum) {
		DataSource dataSource = new GenericDataSource("split", new TextEventDAO(new File(DATASET), eventFormat), new PreferenceDomain(MIN, MAX));
		ItemEventDAO idao = dataSource.getItemEventDAO();
		ScoredItemAccumulator accum = new TopNScoredItemAccumulator(popNum);
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

	private static void setParameters() {
		Properties prop = new Properties();
		InputStream input = null;

		try {
			input = new FileInputStream("config.properties");
			prop.load(input);

			DATASET = (String) prop.get("dataset");
			System.out.println("dataset " + DATASET);

			DATASET_CONTENT = (String) prop.get("dataset_content");
			System.out.println("dataset_content " + DATASET_CONTENT);

			MIN = Double.valueOf((String) prop.get("min_rating"));
			System.out.println("min_rating " + MIN);

			MAX = Double.valueOf((String) prop.get("max_rating"));
			System.out.println("max_rating " + MAX);

			CROSSFOLD_NUMBER = Integer.valueOf((String) prop.get("crossfold"));
			System.out.println("crossfold " + CROSSFOLD_NUMBER);

			HOLDOUT_NUMBER = Integer.valueOf((String) prop.get("holdout"));
			System.out.println("holdout " + HOLDOUT_NUMBER);

			SHORT_HEAD_END = Integer.valueOf((String) prop.get("short_head_end"));
			System.out.println("short_head_end " + SHORT_HEAD_END);

			LONG_TAIL_START = Integer.valueOf((String) prop.get("long_tail_start"));
			System.out.println("long_tail_start " + LONG_TAIL_START);

			POPULAR_ITEMS_SERENDIPITY_NUMBER = Integer.valueOf((String) prop.get("popular_items_number"));
			System.out.println("popular_items_number " + POPULAR_ITEMS_SERENDIPITY_NUMBER);

			RANDOM_ITEMS_FOR_CANDIDATES = Integer.valueOf((String) prop.get("random_items_candidates"));
			System.out.println("random_items_candidates " + RANDOM_ITEMS_FOR_CANDIDATES);

			POPULAR_ITEMS_FOR_CANDIDATES = Integer.valueOf((String) prop.get("popular_items_candidates"));
			System.out.println("popular_items_candidates " + POPULAR_ITEMS_FOR_CANDIDATES);

			TRAIN_TEST_FOLDER_NAME = (String) prop.get("train_folder");
			System.out.println("train_folder " + TRAIN_TEST_FOLDER_NAME);

			OUTPUT_PATH = (String) prop.get("output");
			System.out.println("output " + OUTPUT_PATH);

			OUTPUT_USER_PATH = (String) prop.get("output_user");
			System.out.println("output_user " + OUTPUT_USER_PATH);

			OUTPUT_ITEM_PATH = (String) prop.get("outout_item");
			System.out.println("outout_item " + OUTPUT_ITEM_PATH);

			AlgorithmUtil.THRESHOLD = Double.valueOf((String) prop.get("threshold"));
			System.out.println("threshold " + AlgorithmUtil.THRESHOLD);

			AlgorithmUtil.FEATURE_COUNT = Integer.valueOf((String) prop.get("feature_count"));
			System.out.println("feature_count " + AlgorithmUtil.FEATURE_COUNT);

			AlgorithmUtil.ITERATION_COUNT = Integer.valueOf((String) prop.get("iteration_count"));
			System.out.println("iteration_count " + AlgorithmUtil.ITERATION_COUNT);

			AlgorithmUtil.LEARNING_RATE = Double.valueOf((String) prop.get("learning_rate"));
			System.out.println("learning_rate " + AlgorithmUtil.LEARNING_RATE);

			AlgorithmUtil.REGULARIZATION_TERM = Double.valueOf((String) prop.get("regularization_term"));
			System.out.println("regularization_term " + AlgorithmUtil.REGULARIZATION_TERM);

			AlgorithmUtil.ZHENG_LEARNING_RATE = Double.valueOf((String) prop.get("zheng_learning_rate"));
			System.out.println("zheng_learning_rate " + AlgorithmUtil.ZHENG_LEARNING_RATE);

			AlgorithmUtil.ZHENG_REGULARIZATION_TERM = Double.valueOf((String) prop.get("zheng_regularization_term"));
			System.out.println("zheng_regularization_term " + AlgorithmUtil.ZHENG_REGULARIZATION_TERM);

			AlgorithmUtil.ALPHA = Double.valueOf((String) prop.get("alpha"));
			System.out.println("alpha " + AlgorithmUtil.ALPHA);

			AlgorithmUtil.LU_LEARNING_RATE = Double.valueOf((String) prop.get("lu_learning_rate"));
			System.out.println("lu_learning_rate " + AlgorithmUtil.LU_LEARNING_RATE);

			AlgorithmUtil.LU_REGULARIZATION_TERM = Double.valueOf((String) prop.get("lu_regularization_term"));
			System.out.println("lu_regularization_term " + AlgorithmUtil.LU_REGULARIZATION_TERM);

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

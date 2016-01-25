import evaluationMetric.PopSerendipityTopNMetric;
import evaluationMetric.SerendipityTopNMetric;
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

import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public class SmallExperimentRunner {
	private static final int CROSSFOLD_NUMBER = 10;
	private static final int HOLDOUT_NUMBER = 10;
	private static final int AT_N = 30;
	private static final int START_AT_N = 5;
	private static final int EXPECTED_ITEMS_NUMBER = 9;
	private static final double THRESHOLD = 3.0;
	private static final int POPULAR_ITEMS_NUMBER = 50;
	private static final String DATASET = "D:\\bigdata\\movielens\\ml-100k\\u.data";
	private static final String DATASET_CONTENT = "D:\\bigdata\\movielens\\ml-100k\\small_content.dat";
	private static final String TRAIN_TEST_FOLDER_NAME = "task";
	private static final String OUTPUT_PATH = "/out.csv";
	private static final String OUTPUT_USER_PATH = "/user.csv";
	private static final String OUTPUT_ITEM_PATH = "/item.csv";

	private static final double MIN = 0;
	private static final double MAX = 5;

	private static DelimitedColumnEventFormat eventFormat;
	private static Map<Long, SparseVector> itemContentMap;

	private static final int FEATURE_COUNT = 10;

	private static void setEvaluator(SimpleEvaluator evaluator) {
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

	public static void main(String args[]) {
		SimpleEvaluator evaluator = new SimpleEvaluator();
		setEvaluator(evaluator);

		Map<String, LenskitConfiguration> configurationMap = AlgorithmUtil.getMap(FEATURE_COUNT);

		if (args[0].equals("all")) {
			System.out.println("all");
			for (Map.Entry<String, LenskitConfiguration> entry : configurationMap.entrySet()) {
				evaluator.addAlgorithm(entry.getKey(), entry.getValue());
			}
		} else {
			System.out.println("Algorithms");
			for (String alg : args) {
				System.out.println(alg);
				if (!configurationMap.containsKey(alg)) {
					System.out.println(alg + " doesn't exist");
					System.exit(1);
				}
				evaluator.addAlgorithm(alg, configurationMap.get(alg));
			}
		}


		itemContentMap = ContentUtil.getItemContentMap(DATASET_CONTENT);

		addEvaluationMetrics(evaluator);

		try {
			evaluator.call();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void addEvaluationMetrics(SimpleEvaluator evaluator) {
		ItemSelector popCandidates = ItemSelectors.union(new MyPopularItemSelector(getPopItems()), ItemSelectors.testItems());
		//addMetricsWithParameters(evaluator, AT_N, popCandidates, POPULAR_ITEMS_NUMBER + "pop");

		addMetricsWithParameters(evaluator, AT_N, ItemSelectors.allItems(), "all");

		//addMetricsWithParameters(evaluator, START_AT_N, ItemSelectors.testItems(), "test");

		//addMetricsWithParameters(evaluator, AT_N, ItemSelectors.union(ItemSelectors.testItems(), ItemSelectors.nRandom(POPULAR_ITEMS_NUMBER)), POPULAR_ITEMS_NUMBER + "rand");
	}

	private static void addMetricsWithParameters(SimpleEvaluator evaluator, int maxNumber, ItemSelector candidates, String prefix) {
		ItemSelector threshold = ItemSelectors.testRatingMatches(Matchers.greaterThan(THRESHOLD));
		ItemSelector exclude = ItemSelectors.trainingItems();
		for (int i = START_AT_N; i <= maxNumber; i += 5) {
			String suffix = prefix + "." + i;
			evaluator.addMetric(new PrecisionRecallTopNMetric("", suffix, i, candidates, exclude, threshold));
			evaluator.addMetric(new NDCGTopNMetric("", suffix, i, candidates, exclude));
			evaluator.addMetric(new PopSerendipityTopNMetric(suffix, i, EXPECTED_ITEMS_NUMBER, candidates, exclude, threshold));
			evaluator.addMetric(new SerendipityTopNMetric("content." + suffix, i, EXPECTED_ITEMS_NUMBER, candidates, exclude, threshold, itemContentMap, 300));
		}
	}

	private static LongSet getPopItems() {
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

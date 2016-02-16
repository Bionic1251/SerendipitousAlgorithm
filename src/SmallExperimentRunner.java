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
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors;
import org.grouplens.lenskit.eval.traintest.SimpleEvaluator;
import org.grouplens.lenskit.util.ScoredItemAccumulator;
import org.grouplens.lenskit.util.TopNScoredItemAccumulator;
import org.grouplens.lenskit.vectors.SparseVector;
import org.hamcrest.Matchers;
import util.AlgorithmUtil;
import util.ContentUtil;
import util.MyPopularItemSelector;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Map;

public class SmallExperimentRunner {
	private static final int CROSSFOLD_NUMBER = 1;
	private static final int HOLDOUT_NUMBER = 15;
	private static final int AT_N = 30;
	private static final int START_AT_N = 1;
	private static final int SHORT_HEAD_END = 9;
	private static final int LONG_TAIL_START = 79;
	private static final int POPULAR_ITEMS_NUMBER = 22;
	private static final int POPULAR_ITEMS_FOR_CANDIDATES = 50;
	private static final int POPULAR_ITEMS_FOR_CANDIDATES2 = 30;
	private static final int POPULAR_ITEMS_FOR_CANDIDATES3 = 200;
	private static final double THRESHOLD = 3.0;
	private static final String DATASET = "ml/small/ratings.dat";
	private static final String DATASET_CONTENT = "ml/small/content.dat";
	private static final String TRAIN_TEST_FOLDER_NAME = "task";
	private static final String OUTPUT_PATH = "/out.csv";
	private static final String OUTPUT_USER_PATH = "/user.csv";
	private static final String OUTPUT_ITEM_PATH = "/item.csv";

	private static final double MIN = 0;
	private static final double MAX = 5;

	private static DelimitedColumnEventFormat eventFormat;
	private static Map<Long, SparseVector> itemContentMap;

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

		Map<String, LenskitConfiguration> configurationMap = AlgorithmUtil.getMap();

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
		/*ItemSelector popCandidates = ItemSelectors.union(new MyPopularItemSelector(getPopItems(POPULAR_ITEMS_FOR_CANDIDATES)), ItemSelectors.testItems());
		addMetricsWithParameters(evaluator, popCandidates, POPULAR_ITEMS_FOR_CANDIDATES + "pop");*/

/*		popCandidates = ItemSelectors.union(new MyPopularItemSelector(getPopItems(POPULAR_ITEMS_FOR_CANDIDATES2)), ItemSelectors.testItems());
		addMetricsWithParameters(evaluator, popCandidates, POPULAR_ITEMS_FOR_CANDIDATES2 + "pop");*/

		ItemSelector popCandidates = ItemSelectors.union(new MyPopularItemSelector(getPopItems(POPULAR_ITEMS_FOR_CANDIDATES3)), ItemSelectors.testItems());
		//addMetricsWithParameters(evaluator, popCandidates, POPULAR_ITEMS_FOR_CANDIDATES3 + "pop");

		//addMetricsWithParameters(evaluator, ItemSelectors.allItems(), "all");

		addMetricsWithParameters(evaluator, ItemSelectors.testItems(), "test");

		//addMetricsWithParameters(evaluator, ItemSelectors.union(ItemSelectors.testItems(), ItemSelectors.nRandom(POPULAR_ITEMS_FOR_CANDIDATES2)), POPULAR_ITEMS_FOR_CANDIDATES2 + "rand");
	}

	private static void addMetricsWithParameters(SimpleEvaluator evaluator, ItemSelector candidates, String prefix) {
		ItemSelector threshold = ItemSelectors.testRatingMatches(Matchers.greaterThan(THRESHOLD));
		ItemSelector exclude = ItemSelectors.trainingItems();
		evaluator.addMetric(new AggregatePrecisionRecallTopNMetric(prefix, "", candidates, exclude, threshold));
		evaluator.addMetric(new AggregateNDCGTopNMetric(prefix, "", candidates, exclude));
		evaluator.addMetric(new AggregatePopSerendipityTopNMetric(prefix, POPULAR_ITEMS_NUMBER, candidates, exclude, threshold));
		evaluator.addMetric(new AggregateSerendipityTopNMetric("content." + prefix, SHORT_HEAD_END, candidates, exclude, threshold, itemContentMap, LONG_TAIL_START));
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
}

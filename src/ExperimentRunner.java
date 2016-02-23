import evaluationMetric.*;
import it.unimi.dsi.fastutil.longs.LongSet;
import org.grouplens.lenskit.core.LenskitConfiguration;
import org.grouplens.lenskit.cursors.Cursor;
import org.grouplens.lenskit.data.dao.ItemEventDAO;
import org.grouplens.lenskit.data.event.Event;
import org.grouplens.lenskit.data.history.ItemEventCollection;
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
import org.hamcrest.Matchers;
import util.*;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Map;
import java.util.Properties;

public class ExperimentRunner {

	private static DelimitedColumnEventFormat eventFormat;

	private static void setEvaluator(SimpleEvaluator evaluator) {
		eventFormat = new DelimitedColumnEventFormat(new RatingEventType());
		DataSource dataSource = new GenericDataSource("split", new TextEventDAO(new File(Settings.DATASET), eventFormat));
		CrossfoldTask task = new CrossfoldTask(Settings.TRAIN_TEST_FOLDER_NAME);
		task.setHoldout(Settings.HOLDOUT_NUMBER);
		task.setPartitions(Settings.CROSSFOLD_NUMBER);
		task.setSource(dataSource);
		evaluator.addDataset(task);

		Date cur = new Date();
		SimpleDateFormat format = new SimpleDateFormat("dd.MM.yy_HH.mm.ss");

		evaluator.setOutputPath("out/" + format.format(cur) + Settings.OUTPUT_PATH);
		evaluator.setUserOutputPath("out/" + format.format(cur) + Settings.OUTPUT_USER_PATH);
		evaluator.setPredictOutputPath("out/" + format.format(cur) + Settings.OUTPUT_ITEM_PATH);
	}

	public static void main(String algs[]) {
		setParameters();
		SimpleEvaluator evaluator = new SimpleEvaluator();
		setEvaluator(evaluator);

		addAlgorithms(algs, evaluator);

		ContentAverageDissimilarity.create(Settings.DATASET_CONTENT);

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

		//addMetricsWithParameters(evaluator, ItemSelectors.allItems(), "all");

		ItemSelector popCandidates = ItemSelectors.union(new MyPopularItemSelector(getPopItems(Settings.POPULAR_ITEMS_FOR_CANDIDATES)), ItemSelectors.testItems());
		//addMetricsWithParameters(evaluator, popCandidates, POPULAR_ITEMS_FOR_CANDIDATES + "pop");

		//addMetricsWithParameters(evaluator, ItemSelectors.union(ItemSelectors.testItems(), ItemSelectors.nRandom(RANDOM_ITEMS_FOR_CANDIDATES)), RANDOM_ITEMS_FOR_CANDIDATES + "rand");
	}

	private static void addMetricsWithParameters(SimpleEvaluator evaluator, ItemSelector candidates, String prefix) {
		ItemSelector threshold = ItemSelectors.testRatingMatches(Matchers.greaterThan(Settings.R_THRESHOLD));
		ItemSelector exclude = ItemSelectors.trainingItems();
		evaluator.addMetric(new AggregatePrecisionRecallTopNMetric(prefix, "", candidates, exclude, threshold));
		evaluator.addMetric(new AggregateNDCGTopNMetric(prefix, "", candidates, exclude));
		evaluator.addMetric(new AggregatePopSerendipityTopNMetric(prefix, Settings.POPULAR_ITEMS_SERENDIPITY_NUMBER, candidates, exclude, threshold));
		evaluator.addMetric(new AggregateSerendipityNDCGMetric("RANK22" + prefix, "", candidates, exclude, Settings.R_THRESHOLD,
				Settings.U_THRESHOLD, Settings.D_THRESHOLD));
	}

	private static LongSet getPopItems(int popNum) {
		DataSource dataSource = new GenericDataSource("split", new TextEventDAO(new File(Settings.DATASET), eventFormat));
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

			Settings.ITERATION_COUNT = Integer.valueOf((String) prop.get("iteration_count"));
			System.out.println("iteration_count " + Settings.ITERATION_COUNT);

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

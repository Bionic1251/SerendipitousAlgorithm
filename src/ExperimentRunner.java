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
import org.grouplens.lenskit.eval.metrics.predict.RMSEPredictMetric;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors;
import org.grouplens.lenskit.eval.traintest.SimpleEvaluator;
import org.grouplens.lenskit.util.ScoredItemAccumulator;
import org.grouplens.lenskit.util.TopNScoredItemAccumulator;
import org.hamcrest.Matchers;
import util.*;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Map;

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
		Util.setParameters();
		SimpleEvaluator evaluator = new SimpleEvaluator();
		setEvaluator(evaluator);

		addAlgorithms(algs, evaluator);
		//addDiversification(evaluator);

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

	private static void addDiversification(SimpleEvaluator evaluator) {
		for (int factor = 0; factor < 11; factor += 1) {
			evaluator.addAlgorithm("TDAAdvSVD" + factor, AlgorithmUtil.getTDAAdvancedSVD(Settings.DIVERSIFICATION_FACTOR, factor / 10.0));
			evaluator.addAlgorithm("TDAGenreSVD" + factor, AlgorithmUtil.getTDAGenreSVD(Settings.DIVERSIFICATION_FACTOR, factor / 10.0));
		}
	}

	private static void addEvaluationMetrics(SimpleEvaluator evaluator) {
		//evaluator.addMetric(new RMSEPredictMetric());
		addMetricsWithParameters(evaluator, ItemSelectors.allItems(), "all");

		addMetricsWithParameters(evaluator, ItemSelectors.testItems(), "test");

		//addOnePlusRandomMetric(evaluator);
		//ItemSelector popCandidates = ItemSelectors.union(new MyPopularItemSelector(getPopItems(Settings.POPULAR_ITEMS_FOR_CANDIDATES)), ItemSelectors.testItems());
		//addMetricsWithParameters(evaluator, popCandidates, POPULAR_ITEMS_FOR_CANDIDATES + "pop");
		//addMetricsWithParameters(evaluator, ItemSelectors.union(ItemSelectors.testItems(), ItemSelectors.nRandom(RANDOM_ITEMS_FOR_CANDIDATES)), RANDOM_ITEMS_FOR_CANDIDATES + "rand");
	}

	private static void addMetricsWithParameters(SimpleEvaluator evaluator, ItemSelector candidates, String prefix) {
		ItemSelector threshold = ItemSelectors.testRatingMatches(Matchers.greaterThan(Settings.R_THRESHOLD));
		ItemSelector exclude = ItemSelectors.trainingItems();
		evaluator.addMetric(new AggregatePrecisionRecallTopNMetric(prefix, "", candidates, exclude, threshold));
		evaluator.addMetric(new AggregateNDCGTopNMetric(prefix, "", candidates, exclude));
		evaluator.addMetric(new AggregateNewNRDUMetric("RANK22" + prefix, "", candidates, exclude, Settings.R_THRESHOLD,
				Settings.U_THRESHOLD, Settings.D_THRESHOLD));
		evaluator.addMetric(new AggregatePopSerendipityTopNMetric(prefix, Settings.POPULAR_ITEMS_SERENDIPITY_NUMBER, candidates, exclude, threshold));
		evaluator.addMetric(new AggregateGenresSerendipityTopNMetric(prefix, Settings.POPULAR_ITEMS_SERENDIPITY_NUMBER, candidates, exclude));
		evaluator.addMetric(new AggregateDiversityMetric(prefix, candidates, exclude));
		//evaluator.addMetric(new AggregateCustomSerendipityTopNMetric(prefix, Settings.POPULAR_ITEMS_SERENDIPITY_NUMBER, candidates, exclude, threshold));
		//evaluator.addMetric(new WriterMetric());
		//evaluator.addMetric(new AggregatePrecisionRecallTopNMetric(prefix, "", candidates, exclude, threshold));
		/*evaluator.addMetric(new AggregateNRDUMetric("RANK22" + prefix, "", candidates, exclude, Settings.R_THRESHOLD,
				Settings.U_THRESHOLD, Settings.D_THRESHOLD));*/
		//evaluator.addMetric(new AggreagateComponentMetric(prefix, candidates, exclude));
	}

	private static void addOnePlusRandomMetric(SimpleEvaluator evaluator) {
		ItemSelector threshold = ItemSelectors.testRatingMatches(Matchers.greaterThanOrEqualTo(5.0));
		ItemSelector exclude = ItemSelectors.trainingItems();
		ItemSelector testTrain = ItemSelectors.union(ItemSelectors.trainingItems(), ItemSelectors.testItems());
		ItemSelector randomItems = ItemSelectors.randomSubset(ItemSelectors.setDifference(ItemSelectors.allItems(), testTrain), 1000);
		ItemSelector randCandidates = ItemSelectors.union(randomItems, ItemSelectors.testItems());
		evaluator.addMetric(new AggregateOPRMetric(randCandidates, exclude, threshold));
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
}

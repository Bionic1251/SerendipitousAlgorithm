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

public class NewExperimentRunner {

	private static DelimitedColumnEventFormat eventFormat;

	private static void setEvaluator(SimpleEvaluator evaluator) {
		eventFormat = new DelimitedColumnEventFormat(new RatingEventType());
		DataSource train = new GenericDataSource("split", new TextEventDAO(new File(Settings.DATASET), eventFormat));
		DataSource test = new GenericDataSource("split", new TextEventDAO(new File("D:\\gdrive\\PhD stuff\\Research\\Serendipitous algorithm\\results\\OnlineExperiments\\Recommendations\\test.txt"), eventFormat));
		evaluator.addDataset(train, test);

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
		//addOnePlusRandomMetric(evaluator);

		addMetricsWithParameters(evaluator, ItemSelectors.testItems(), "test");

		//addMetricsWithParameters(evaluator, ItemSelectors.allItems(), "all");

		//ItemSelector popCandidates = ItemSelectors.union(new MyPopularItemSelector(getPopItems(Settings.POPULAR_ITEMS_FOR_CANDIDATES)), ItemSelectors.testItems());
		//addMetricsWithParameters(evaluator, popCandidates, POPULAR_ITEMS_FOR_CANDIDATES + "pop");

		//addMetricsWithParameters(evaluator, ItemSelectors.union(ItemSelectors.testItems(), ItemSelectors.nRandom(RANDOM_ITEMS_FOR_CANDIDATES)), RANDOM_ITEMS_FOR_CANDIDATES + "rand");
	}

	private static void addMetricsWithParameters(SimpleEvaluator evaluator, ItemSelector candidates, String prefix) {
		ItemSelector threshold = ItemSelectors.testRatingMatches(Matchers.greaterThan(Settings.R_THRESHOLD));
		ItemSelector exclude = ItemSelectors.trainingItems();
		evaluator.addMetric(new AggregatePrecisionRecallTopNMetric(prefix, "", candidates, exclude, threshold));
		/*evaluator.addMetric(new AggregateNDCGTopNMetric(prefix, "", candidates, exclude));
		evaluator.addMetric(new AggregatePopSerendipityTopNMetric(prefix, Settings.POPULAR_ITEMS_SERENDIPITY_NUMBER, candidates, exclude, threshold));
		evaluator.addMetric(new AggregateNRDUMetric("RANK22" + prefix, "", candidates, exclude, Settings.R_THRESHOLD,
				Settings.U_THRESHOLD, Settings.D_THRESHOLD));
		evaluator.addMetric(new AggreagateComponentMetric(prefix, candidates, exclude));*/
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

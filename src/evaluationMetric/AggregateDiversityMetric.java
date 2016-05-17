package evaluationMetric;

import it.unimi.dsi.fastutil.longs.LongSet;
import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.eval.Attributed;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.AbstractMetric;
import org.grouplens.lenskit.eval.metrics.ResultColumn;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors;
import org.grouplens.lenskit.eval.traintest.TestUser;
import org.grouplens.lenskit.scored.ScoredId;
import org.grouplens.lenskit.util.statistics.MeanAccumulator;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.hamcrest.Matchers;
import util.ContentAverageDissimilarity;
import util.ContentUtil;
import util.PrepareUtil;
import util.Settings;

import java.util.*;


public class AggregateDiversityMetric extends AbstractMetric<MeanAccumulator, AggregateDiversityMetric.AggregateResult, AggregateDiversityMetric.AggregateResult> {
	private final ItemSelector candidates;
	private final ItemSelector exclude;
	private final String suffix;
	private String dataSetName;

	private MeanAccumulator context1;
	private MeanAccumulator context5;
	private MeanAccumulator context10;
	private MeanAccumulator context15;
	private MeanAccumulator context20;
	private MeanAccumulator context25;
	private MeanAccumulator context30;

	public AggregateDiversityMetric(String suffix, ItemSelector candidates, ItemSelector exclude) {
		super(AggregateResult.class, AggregateResult.class);
		this.suffix = suffix;
		this.candidates = candidates;
		this.exclude = exclude;
	}


	@Override
	protected String getSuffix() {
		return suffix;
	}

	@Override
	protected AggregateResult doMeasureUser(TestUser user, MeanAccumulator context) {
		List<ScoredId> recommendations = user.getRecommendations(30, candidates, exclude);
		if (recommendations == null || recommendations.isEmpty()) {
			context1.add(0.0);
			context5.add(0.0);
			context10.add(0.0);
			context15.add(0.0);
			context20.add(0.0);
			context25.add(0.0);
			context30.add(0.0);
			return null;
		}
		double ser1 = measureUser(user, context1, recommendations, 1);
		double ser5 = measureUser(user, context5, recommendations, 5);
		double ser10 = measureUser(user, context10, recommendations, 10);
		double ser15 = measureUser(user, context15, recommendations, 15);
		double ser20 = measureUser(user, context20, recommendations, 20);
		double ser25 = measureUser(user, context25, recommendations, 25);
		double ser30 = measureUser(user, context30, recommendations, 30);
		return new AggregateResult(ser1, ser5, ser10, ser15, ser20, ser25, ser30, 1);
	}

	protected double measureUser(TestUser user, MeanAccumulator context, List<ScoredId> recommendations, int listSize) {
		if (recommendations.size() > listSize) {
			recommendations = new ArrayList<ScoredId>(recommendations.subList(0, listSize));
		}

		double sum = 0;
		for (ScoredId outerRec : recommendations) {
			for (ScoredId innerRec : recommendations) {
				if (innerRec.getId() == outerRec.getId()) {
					continue;
				}
				sum += getDissimilarity(innerRec.getId(), outerRec.getId());
			}
		}

		double val = sum / (recommendations.size() * (recommendations.size() - 1));
		context.add(val);
		return val;
	}

	private double getDissimilarity(Long itemId1, Long itemId2) {
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> itemMap = dissimilarity.getItemContentMap();
		if (!itemMap.containsKey(itemId1) || !itemMap.containsKey(itemId2)) {
			return 1;
		}
		SparseVector vec1 = itemMap.get(itemId1);
		//vec1 = dissimilarity.toTFIDF(vec1);
		SparseVector vec2 = itemMap.get(itemId2);
		//vec2 = dissimilarity.toTFIDF(vec2);
		double sim = ContentUtil.getJaccard(vec1, vec2);
		return 1 - sim;
	}

	@Override
	protected AggregateResult getTypedResults(MeanAccumulator context) {
		long count = context30.getCount();
		double ser1 = context1.getMean();
		double ser5 = context5.getMean();
		double ser10 = context10.getMean();
		double ser15 = context15.getMean();
		double ser20 = context20.getMean();
		double ser25 = context25.getMean();
		double ser30 = context30.getMean();
		return new AggregateResult(ser1, ser5, ser10, ser15, ser20, ser25, ser30, count);
	}

	@Override
	public MeanAccumulator createContext(Attributed algorithm, TTDataSet dataSet, Recommender recommender) {
		context1 = new MeanAccumulator();
		context5 = new MeanAccumulator();
		context10 = new MeanAccumulator();
		context15 = new MeanAccumulator();
		context20 = new MeanAccumulator();
		context25 = new MeanAccumulator();
		context30 = new MeanAccumulator();
		//updateExpectedItems(dataSet);
		//updatePopMap(dataSet);
		dataSetName = dataSet.getTrainingData().getName();
		//getRatingThresholds(dataSet);
		return new MeanAccumulator();
	}

	public static class AggregateResult {
		@ResultColumn("Diversity1")
		public final double ser1;

		@ResultColumn("Diversity5")
		public final double ser5;

		@ResultColumn("Diversity10")
		public final double ser10;

		@ResultColumn("Diversity15")
		public final double ser15;

		@ResultColumn("Diversity20")
		public final double ser20;

		@ResultColumn("Diversity25")
		public final double ser25;

		@ResultColumn("Diversity30")
		public final double ser301;

		@ResultColumn("DiversityCount")
		public final long count;

		public AggregateResult(double ser1, double ser5, double ser10, double ser15, double ser20, double ser25, double ser301, long count) {
			this.ser1 = ser1;
			this.ser5 = ser5;
			this.ser10 = ser10;
			this.ser15 = ser15;
			this.ser20 = ser20;
			this.ser25 = ser25;
			this.ser301 = ser301;
			this.count = count;
		}
	}
}

package lc.investigation;

import it.unimi.dsi.fastutil.longs.LongCollection;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import pop.PopModel;
import util.AverageAggregate;
import util.ContentAverageDissimilarity;

import javax.annotation.Nonnull;
import java.util.Map;

public abstract class InvestigationItemScorer extends AbstractItemScorer {
	protected final PopModel popModel;
	protected final PreferenceSnapshot snapshot;
	protected final Map<Long, SparseVector> userItemDissimilarityMap;
	protected final Map<Long, AverageAggregate> userThresholdMap;

	protected static final double LEARNING_RATE = 0.0001;
	protected static final double P_COEFF = 0.001;
	protected final int iterationCount;
	protected static final double DEFAULT_VAL = 0.33;

	public InvestigationItemScorer(PopModel popModel, PreferenceSnapshot snapshot, int ic) {
		init();
		iterationCount = ic;
		this.popModel = popModel;
		this.snapshot = snapshot;
		ContentAverageDissimilarity contentAverageDissimilarity = ContentAverageDissimilarity.getInstance();
		userItemDissimilarityMap = contentAverageDissimilarity.getUserItemAvgDistanceMap(snapshot);
		userThresholdMap = contentAverageDissimilarity.getAverageMap(snapshot, popModel, userItemDissimilarityMap);

		train();
	}

	protected void init() {
	}

	protected void train() {
		System.out.println(InvestigationItemScorer.class);
		for (int i = 0; i < iterationCount; i++) {
			System.out.println("Iteration " + i);
			printWeights();
			printFunctionForEachUser();
			trainParameters();
		}
	}

	protected abstract void printWeights();

	private void trainParameters() {
		LongCollection userIds = snapshot.getUserIds();
		for (long userId : userIds) {
			trainForEachUser(userId);
		}
	}

	protected abstract void trainForEachUser(long userId);

	protected boolean isSerendipitous(Triple triple, AverageAggregate aggregate) {
		if (triple.rating <= aggregate.getR().getThreshold()) {
			return false;
		}
		if (triple.dissimilarity <= aggregate.getD().getThreshold()) {
			return false;
		}
		if (triple.unpopularity <= aggregate.getU().getThreshold()) {
			return false;
		}
		return true;
	}

	protected double getDissimilarity(long itemId, long userId) {
		if (!userItemDissimilarityMap.containsKey(userId)) {
			return 1;
		}
		SparseVector vector = userItemDissimilarityMap.get(userId);
		if (!vector.containsKey(itemId)) {
			return 1;
		}
		return vector.get(itemId);
	}

	protected void printFunctionForEachUser() {
		LongCollection userIds = snapshot.getUserIds();
		double sum = 0;
		for (long userId : userIds) {
			sum += getFunctionVal(userId);
		}
		System.out.println("Function " + sum);
	}

	protected abstract double getFunctionVal(long userId);

	protected class Triple {
		protected double rating;
		protected double dissimilarity;
		protected double unpopularity;

		protected Triple(long userId, long itemId, double rating, AverageAggregate aggregate) {
			this.rating = aggregate.getR().getNormalizer().norm(rating);
			dissimilarity = aggregate.getD().getNormalizer().norm(getDissimilarity(itemId, userId));
			unpopularity = aggregate.getU().getNormalizer().norm(1 - (double) popModel.getPop(itemId) / popModel.getMax());
		}
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
	}
}

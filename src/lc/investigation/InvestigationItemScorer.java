package lc.investigation;

import it.unimi.dsi.fastutil.longs.LongCollection;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import pop.PopModel;
import util.AverageAggregate;
import util.ContentAverageDissimilarity;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.PrintWriter;
import java.util.Collection;
import java.util.Map;

public abstract class InvestigationItemScorer extends AbstractItemScorer {
	protected double serR;
	protected double serD;
	protected double serU;
	protected double unserR;
	protected double unserD;
	protected double unserU;

	protected final PopModel popModel;
	protected final PreferenceSnapshot snapshot;
	protected final Map<Long, SparseVector> userItemDissimilarityMap;
	protected final Map<Long, AverageAggregate> userThresholdMap;
	private PrintWriter printWriter;

	public InvestigationItemScorer(PopModel popModel, PreferenceSnapshot snapshot, String filename) {
		this.popModel = popModel;
		this.snapshot = snapshot;
		ContentAverageDissimilarity contentAverageDissimilarity = ContentAverageDissimilarity.getInstance();
		userItemDissimilarityMap = contentAverageDissimilarity.getUserItemAvgDistanceMap(snapshot);
		userThresholdMap = contentAverageDissimilarity.getAverageMap(snapshot, popModel, userItemDissimilarityMap);
		try {
			printWriter = new PrintWriter(new File(filename));
			printWriter.println("userId\tprofileSize\twr\twd\twu\tserR\tserD\tserU\tunserR\tunserD\tunserU");
		} catch (Exception e) {
			e.printStackTrace();
		}

		train();
	}

	protected void print(long userId, int profileSize, WeightTriple triple, double serR, double serD, double serU, double unserR, double unserD, double unserU) {
		String s = "\t";
		printWriter.println(userId + s + profileSize + s + triple.getWr() + s + triple.getWd() + s + triple.getWu() + s + serR + s + serD + s + serU + s + unserR + s + unserD + s + unserU);
	}

	protected void close() {
		printWriter.close();
	}

	protected void train() {
		LongCollection userIds = snapshot.getUserIds();
		for (long userId : userIds) {
			countForEachUser(userId);
		}
	}

	protected void countForEachUser(Long userId) {
		Collection<IndexedPreference> prefs = snapshot.getUserRatings(userId);
		AverageAggregate aggregate = userThresholdMap.get(userId);
		for (IndexedPreference innerPref : prefs) {
			for (IndexedPreference outerPref : prefs) {
				if (innerPref.getItemId() == outerPref.getItemId()) {
					continue;
				}
				Triple innerTriple = new Triple(userId, innerPref.getItemId(), innerPref.getValue(), aggregate);
				Triple outerTriple = new Triple(userId, outerPref.getItemId(), outerPref.getValue(), aggregate);
				boolean innerSer = isSerendipitous(innerTriple, aggregate);
				boolean outerSer = isSerendipitous(outerTriple, aggregate);
				if (!innerSer && outerSer) {
					sumParameters(outerTriple, innerTriple);
				} else {
					if (innerSer && !outerSer) {
						sumParameters(innerTriple, outerTriple);
					}
				}
			}
		}
	}

	private void sumParameters(Triple serTriple, Triple unserTriple) {
		serR += serTriple.rating;
		serD += serTriple.dissimilarity;
		serU += serTriple.unpopularity;
		unserR += unserTriple.rating;
		unserD += unserTriple.dissimilarity;
		unserU += unserTriple.unpopularity;
	}

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

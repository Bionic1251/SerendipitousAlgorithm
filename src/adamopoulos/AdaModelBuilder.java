package adamopoulos;

import annotation.RatingPredictor;
import annotation.Threshold;
import it.unimi.dsi.fastutil.longs.LongCollection;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.knn.item.model.ItemItemModel;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import pop.PopModel;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;

public class AdaModelBuilder implements Provider<AdaModel> {
	private final ItemScorer baseline;
	private final ItemItemModel itemItemModel;
	private final PopModel popModel;
	private final PreferenceSnapshot snapshot;
	private final double threshold;

	@Inject
	public AdaModelBuilder(@RatingPredictor ItemScorer baseline, ItemItemModel itemItemModel, PopModel popModel,
						   @Transient @Nonnull PreferenceSnapshot snapshot, @Threshold double threshold) {
		System.out.println("AdaModelBuilder");
		this.baseline = baseline;
		this.itemItemModel = itemItemModel;
		this.snapshot = snapshot;
		this.threshold = threshold;
		this.popModel = popModel;
	}

	@Override
	public AdaModel get() {
		AdaModel model = createModel();
		//countError(model);
		//for (int i = 0; i < 2; i++) {
			learnParameters(model);
			//countError(model);
		//}
		return model;
	}

	private void countError(AdaModel model) {
		System.out.println("Error calculation");
		LongCollection items = snapshot.getItemIds();
		LongCollection users = snapshot.getUserIds();
		int count = 0;
		int error = 0;
		for (Long userId : users) {
			count++;
			if (count % 100 == 0) {
				System.out.println(count + " users processed");
			}
			SparseVector prediction = baseline.score(userId, items);
			Collection<IndexedPreference> preferences = snapshot.getUserRatings(userId);
			for (IndexedPreference rating1 : preferences) {
				for (IndexedPreference rating2 : preferences) {
					int serendipity1 = getSerendipity(rating1);
					int serendipity2 = getSerendipity(rating2);
					if (rating1.equals(rating2) || serendipity1 == 0 && serendipity2 == 0) {
						continue;
					}

					double d1 = model.getDistance(userId, rating1.getItemId());
					double d2 = model.getDistance(userId, rating2.getItemId());
					double r1 = prediction.get(rating1.getItemId());
					double r2 = prediction.get(rating2.getItemId());
					double rank1 = model.getRank(r1, d1);
					double rank2 = model.getRank(r2, d2);
					if (((serendipity1 > serendipity2 && rank1 < rank2) || (serendipity2 > serendipity1 && rank2 < rank1)) && d1 != d2) {
						error++;
					}
				}
			}
		}
		System.out.println("Error " + error);
	}

	private void learnParameters(AdaModel model) {
		System.out.println("Learning parameters");
		LongCollection items = snapshot.getItemIds();
		LongCollection users = snapshot.getUserIds();
		int count = 0;
		for (Long userId : users) {
			count++;
			if (count % 100 == 0) {
				System.out.println(count + " users processed");
			}
			SparseVector prediction = baseline.score(userId, items);
			Collection<IndexedPreference> preferences = snapshot.getUserRatings(userId);
			for (IndexedPreference rating1 : preferences) {
				for (IndexedPreference rating2 : preferences) {
					int serendipity1 = getSerendipity(rating1);
					int serendipity2 = getSerendipity(rating2);
					if (rating1.equals(rating2) || serendipity1 == 0 && serendipity2 == 0) {
						continue;
					}

					double d1 = model.getDistance(userId, rating1.getItemId());
					double d2 = model.getDistance(userId, rating2.getItemId());
					double r1 = prediction.get(rating1.getItemId());
					double r2 = prediction.get(rating2.getItemId());
					double rank1 = model.getRank(r1, d1);
					double rank2 = model.getRank(r2, d2);
					double q = model.getQ();
					double lambda = model.getLambda();
					double learningRate = 0.0001;
					double regularizationTerm = 0.00001;
					if (serendipity1 > serendipity2 && rank1 < rank2) {
						model.setQ(q + (r1 - r2 - regularizationTerm * q) * learningRate);
						model.setLambda(lambda + (d2 - d1 - regularizationTerm * lambda) * learningRate);
					} else if (serendipity2 > serendipity1 && rank2 < rank1) {
						model.setQ(q + (r2 - r1 - regularizationTerm * q) * learningRate);
						model.setLambda(lambda + (d1 - d2 - regularizationTerm * lambda) * learningRate);
					}
				}
			}
		}
		System.out.println("q " + model.getQ() + "; lambda " + model.getLambda());
	}

	private int getSerendipity(IndexedPreference rating) {
		if (rating.getValue() < threshold) {
			return 0;
		}
		return popModel.getPop(rating.getItemId());
	}

	private AdaModel createModel() {
		System.out.println("Dissimilarity calculation");
		LongCollection users = snapshot.getUserIds();
		LongCollection items = snapshot.getItemIds();
		Map<Long, SparseVector> userItemDissimilarity = new HashMap<Long, SparseVector>();
		int count = 0;
		double globalInternalDissimilarity = 0;
		for (Long userId : users) {
			count++;
			if (count % 50 == 0) {
				System.out.println(count + " users processed");
			}
			int internalCount = 0;
			double internalDissimilarity = 0;
			MutableSparseVector itemDissimilarity = MutableSparseVector.create(items);
			Collection<IndexedPreference> preferences = snapshot.getUserRatings(userId);
			Set<Long> ratedItemSet = new HashSet<Long>();
			for (IndexedPreference pref : preferences) {
				ratedItemSet.add(pref.getItemId());
			}
			for (Long itemId : items) {
				double averageDissimilarity = getAverageDissimilarity(itemId, ratedItemSet);
				if (ratedItemSet.contains(itemId)) {
					internalDissimilarity += averageDissimilarity;
					internalCount++;
				}
				itemDissimilarity.set(itemId, averageDissimilarity);
			}
			globalInternalDissimilarity += internalDissimilarity / internalCount;
			userItemDissimilarity.put(userId, itemDissimilarity);
		}
		globalInternalDissimilarity /= users.size();
		return new AdaModel(globalInternalDissimilarity, baseline, userItemDissimilarity);
	}

	private double getAverageDissimilarity(Long itemId, Set<Long> ratedItemSet) {
		int count = 0;
		double dissimilarity = 0;
		for (Long ratedItemId : ratedItemSet) {
			if (itemId.equals(ratedItemId)) {
				continue;
			}
			dissimilarity += getDissimilarity(itemId, ratedItemId);
			count++;
		}
		if (count == 0) {
			return 0;
		}
		return dissimilarity / count;
	}

	private double getDissimilarity(Long itemId1, Long itemId2) {
		SparseVector neighbors = itemItemModel.getNeighbors(itemId1);
		if (!neighbors.containsKey(itemId2)) {
			return 1;
		}
		return 1 - neighbors.get(itemId2);
	}
}

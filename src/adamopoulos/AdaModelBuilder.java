package adamopoulos;

import annotation.R_Threshold;
import annotation.RatingPredictor;
import it.unimi.dsi.fastutil.longs.LongCollection;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.knn.item.model.ItemItemModel;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import pop.PopModel;
import util.ContentAverageDissimilarity;
import util.ContentUtil;
import util.Settings;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;

public class AdaModelBuilder implements Provider<AdaModel> {
	private final ItemScorer baseline;
	private final PopModel popModel;
	private final PreferenceSnapshot snapshot;
	private final double threshold;
	private final double learningRate = 0.0001;
	private final double regularizationTerm = 0.00001;

	@Inject
	public AdaModelBuilder(@RatingPredictor ItemScorer baseline, PopModel popModel,
						   @Transient @Nonnull PreferenceSnapshot snapshot, @R_Threshold double threshold) {
		this.baseline = baseline;
		this.snapshot = snapshot;
		this.threshold = threshold;
		this.popModel = popModel;
	}

	@Override
	public AdaModel get() {
		System.out.println(AdaModelBuilder.class);
		AdaModel model = createModel();
		countError(model);
		for (int i = 0; i < 3; i++) {
			learnParameters(model); //it converges quickly
			countError(model);
		}
		return model;
	}

	private void countError(AdaModel model) {
		System.out.println("Error calculation");
		LongCollection items = snapshot.getItemIds();
		LongCollection users = snapshot.getUserIds();
		int count = 0;
		int error = 0;
		double sum = 0;
		for (Long userId : users) {
			count++;
			if (count % 100 == 0) {
				System.out.println(count + " users processed");
			}
			Collection<IndexedPreference> preferences = snapshot.getUserRatings(userId);
			for (IndexedPreference rating1 : preferences) {
				for (IndexedPreference rating2 : preferences) {
					double unpop1 = popModel.getPop(rating1.getItemId()) / popModel.getMax();
					double dissim1 = model.getDissimilarity(userId, rating1.getItemId());
					double serendipity1 = getSerendipity(rating1.getValue(), dissim1, unpop1);
					double unpop2 = popModel.getPop(rating2.getItemId()) / popModel.getMax();
					double dissim2 = model.getDissimilarity(userId, rating2.getItemId());
					double serendipity2 = getSerendipity(rating2.getValue(), dissim2, unpop2);
					if (rating1.equals(rating2) || serendipity1 == 0 && serendipity2 == 0) {
						continue;
					}

					double d1 = model.getDistance(userId, rating1.getItemId());
					double d2 = model.getDistance(userId, rating2.getItemId());
					double r1 = rating1.getValue();
					double r2 = rating2.getValue();
					double rank1 = model.getRank(r1, d1);
					double rank2 = model.getRank(r2, d2);
					if (serendipity1 > serendipity2 && rank2 > rank1) {
						sum += rank1 - rank2;
					} else if (serendipity2 > serendipity1 && rank1 > rank2) {
						sum += rank2 - rank1;
					}
				}
			}
		}
		System.out.println("Sum " + sum);
	}

	private void learnParameters(AdaModel model) {
		System.out.println("Learning parameters");
		LongCollection users = snapshot.getUserIds();
		int count = 0;
		for (Long userId : users) {
			count++;
			if (count % 100 == 0) {
				System.out.println(count + " users processed");
			}
			Collection<IndexedPreference> preferences = snapshot.getUserRatings(userId);
			for (IndexedPreference rating1 : preferences) {
				for (IndexedPreference rating2 : preferences) {
					double unpop1 = popModel.getPop(rating1.getItemId()) / popModel.getMax();
					double dissim1 = model.getDissimilarity(userId, rating1.getItemId());
					double serendipity1 = getSerendipity(rating1.getValue(), dissim1, unpop1);
					double unpop2 = popModel.getPop(rating2.getItemId()) / popModel.getMax();
					double dissim2 = model.getDissimilarity(userId, rating2.getItemId());
					double serendipity2 = getSerendipity(rating2.getValue(), dissim2, unpop2);
					if (rating1.equals(rating2) || serendipity1 == 0 && serendipity2 == 0) {
						continue;
					}

					double d1 = model.getDistance(userId, rating1.getItemId());
					double d2 = model.getDistance(userId, rating2.getItemId());
					double r1 = rating1.getValue();
					double r2 = rating2.getValue();
					double q = model.getQ();
					double lambda = model.getLambda();
					double rank1 = model.getRank(r1, d1);
					double rank2 = model.getRank(r2, d2);
					if (serendipity1 > serendipity2 && rank2 > rank1) {
						model.setQ(q + (r1 - r2 - regularizationTerm * q) * learningRate);
						model.setLambda(lambda + (d2 - d1 - regularizationTerm * lambda) * learningRate);
					} else if (serendipity2 > serendipity1 && rank1 > rank2) {
						model.setQ(q + (r2 - r1 - regularizationTerm * q) * learningRate);
						model.setLambda(lambda + (d1 - d2 - regularizationTerm * lambda) * learningRate);
					}
				}
			}
		}
		System.out.println("q " + model.getQ() + "; lambda " + model.getLambda());
	}

	private double getSerendipity(double rating, double dissimilarity, double unpopularity) {
		if (rating <= Settings.R_THRESHOLD) {
			return 0;
		}
		if (dissimilarity <= Settings.D_THRESHOLD) {
			return 0;
		}
		if (unpopularity <= Settings.U_THRESHOLD) {
			return 0;
		}
		double result = rating / Settings.MAX + dissimilarity + unpopularity;
		return result;
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
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> itemMap = dissimilarity.getItemContentMap();
		if (!itemMap.containsKey(itemId1) || !itemMap.containsKey(itemId2)) {
			return 1;
		}
		SparseVector vec1 = itemMap.get(itemId1);
		SparseVector vec2 = itemMap.get(itemId2);
		double sim = ContentUtil.getCosine(vec1, vec2);
		return 1 - sim;
	}
}

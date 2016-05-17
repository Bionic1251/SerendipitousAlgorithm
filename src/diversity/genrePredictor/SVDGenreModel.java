package diversity.genrePredictor;

import com.google.common.collect.ImmutableList;
import mikera.matrixx.Matrix;
import mikera.matrixx.impl.ImmutableMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.impl.ImmutableVector;
import org.grouplens.grapht.annotation.DefaultProvider;
import org.grouplens.lenskit.core.Shareable;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.indexes.IdIndexMapping;
import org.grouplens.lenskit.mf.funksvd.FeatureInfo;
import org.grouplens.lenskit.mf.svd.MFModel;

import java.util.List;

/**
 * Model for getFunkSVD recommendation.  This extends the genrePredictor model with clamping functions and
 * information about the training of the features.
 *
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
@DefaultProvider(SVDGenreModelBuilder.class)
@Shareable
public final class SVDGenreModel {
	/*private static final long serialVersionUID = 3L;

	private final List<FeatureInfo> featureInfo;
	private final AVector averageUser;*/
	private final Matrix userFeatures;
	private final Matrix itemFeatures;
	protected final PreferenceSnapshot snapshot;

	public SVDGenreModel(Matrix userFeatures, Matrix itemFeatures, PreferenceSnapshot snapshot) {
		this.userFeatures = userFeatures;
		this.itemFeatures = itemFeatures;
		this.snapshot = snapshot;
		//super(umat, imat, uidx, iidx);

		/*featureInfo = ImmutableList.copyOf(features);

		double[] means = new double[featureCount];
		for (int f = featureCount - 1; f >= 0; f--) {
			means[f] = featureInfo.get(f).getUserAverage();
		}
		averageUser = ImmutableVector.wrap(means);*/
	}

	/**
	 * Get the {@link org.grouplens.lenskit.mf.funksvd.FeatureInfo} for a particular feature.
	 * @param f The feature number.
	 * @return The feature's summary information.
	 */
	/*public FeatureInfo getFeatureInfo(int f) {
		return featureInfo.get(f);
	}

	*/

	/**
	 * Get the metadata about all features.
	 *
	 * @return The feature metadata.
	 *//*
	public List<FeatureInfo> getFeatureInfo() {
		return featureInfo;
	}

	public AVector getAverageUserVector() {
		return averageUser;
	}*/
	public double getPrediction(long userId, long itemId) {
		if (!snapshot.userIndex().containsId(userId)) {
			return 0.0;
		}
		int itemIndex = (int) itemId;
		int userIndex = snapshot.userIndex().getIndex(userId);
		AVector itemVector = itemFeatures.getRow(itemIndex);
		AVector userVector = userFeatures.getRow(userIndex);
		double prediction = itemVector.dotProduct(userVector);
		return prediction;
	}
}

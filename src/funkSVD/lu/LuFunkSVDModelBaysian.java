package funkSVD.lu;

import com.google.common.collect.ImmutableList;
import mikera.matrixx.impl.ImmutableMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.impl.ImmutableVector;
import org.grouplens.grapht.annotation.DefaultProvider;
import org.grouplens.lenskit.core.Shareable;
import org.grouplens.lenskit.indexes.IdIndexMapping;
import org.grouplens.lenskit.mf.funksvd.FeatureInfo;
import org.grouplens.lenskit.mf.svd.MFModel;

import java.util.List;

/**
 * Model for funkSVD recommendation.  This extends the Baseline model with clamping functions and
 * information about the training of the features.
 *
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
@DefaultProvider(LuFunkSVDModelBuilderBaysian.class)
@Shareable
public final class LuFunkSVDModelBaysian extends MFModel {
	private static final long serialVersionUID = 3L;

	private final List<FeatureInfo> featureInfo;
	private final AVector averageUser;

	public LuFunkSVDModelBaysian(ImmutableMatrix umat, ImmutableMatrix imat,
								 IdIndexMapping uidx, IdIndexMapping iidx,
								 List<FeatureInfo> features) {
		super(umat, imat, uidx, iidx);

		featureInfo = ImmutableList.copyOf(features);

		double[] means = new double[featureCount];
		for (int f = featureCount - 1; f >= 0; f--) {
			means[f] = featureInfo.get(f).getUserAverage();
		}
		averageUser = ImmutableVector.wrap(means);
	}

	/**
	 * Get the {@link FeatureInfo} for a particular feature.
	 * @param f The feature number.
	 * @return The feature's summary information.
	 */
	public FeatureInfo getFeatureInfo(int f) {
		return featureInfo.get(f);
	}

	/**
	 * Get the metadata about all features.
	 * @return The feature metadata.
	 */
	public List<FeatureInfo> getFeatureInfo() {
		return featureInfo;
	}

	public AVector getAverageUserVector() {
		return averageUser;
	}
}

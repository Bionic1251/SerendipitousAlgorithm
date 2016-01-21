package adamopoulos;

import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.knn.item.model.ItemItemModel;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;

import javax.annotation.Nonnull;
import javax.inject.Inject;

public class AdaItemScorer extends AbstractItemScorer {
	private final AdaModel adaModel;

	@Inject
	public AdaItemScorer(AdaModel adaModel) {
		this.adaModel = adaModel;
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		for(VectorEntry e : scores.view(VectorEntry.State.EITHER)){
			scores.set(e, adaModel.getRankById(user, e.getKey()));
		}
	}
}

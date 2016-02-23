package pop;

import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.event.Rating;
import org.grouplens.lenskit.data.history.UserHistory;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.List;

public class PopItemScorer extends AbstractItemScorer {
	private PopModel model;

	@Inject
	public PopItemScorer(PopModel model) {
		this.model = model;
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		List<Long> recommendations = model.getItemList();
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			int score = recommendations.indexOf(e.getKey());
			scores.set(e, score);
		}
	}
}

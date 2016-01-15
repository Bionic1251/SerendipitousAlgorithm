package evaluationMetric;

import org.grouplens.lenskit.eval.metrics.predict.NDCGPredictMetric;

import java.util.ArrayList;
import java.util.List;

public class MyMetric extends NDCGPredictMetric {
	@Override
	public List<String> getColumnLabels() {
		List<String> list = new ArrayList<String>();
		list.add("yo");
		return list;
	}
}

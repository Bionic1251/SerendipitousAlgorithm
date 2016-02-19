package recBrowser;

import evaluationMetric.Container;

import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class UserRecReader extends BrowserFileReader {
	private List<Container<Double>> scoreList;
	private boolean firstLine = true;
	private final String algName;
	private final String userId;
	private NumberFormat numberFormat;

	public UserRecReader(String algName, String userId) {
		this.algName = algName;
		this.userId = userId;
		scoreList = new ArrayList<Container<Double>>();
		numberFormat = NumberFormat.getInstance();
		numberFormat.setMaximumFractionDigits(3);
	}

	@Override
	protected void processLine(String line) {
		if (firstLine) {
			firstLine = false;
			return;
		}
		String[] brokenLine = line.split(",");
		String alg = brokenLine[0];
		String user = brokenLine[3];
		if (alg.equals(algName) && user.equals(userId)) {
			String itemId = brokenLine[4];
			String scoreStr = brokenLine[6];
			Long item = Long.valueOf(itemId);
			Double score = Double.valueOf(scoreStr);
			scoreList.add(new Container<Double>(item, score));
		}
	}

	public List<Container<Double>> getScoreList() {
		return scoreList;
	}

	public String[] getRecs() {
		Collections.sort(scoreList);
		Collections.reverse(scoreList);
		String[] array = new String[scoreList.size()];
		for (int i = 0; i < scoreList.size(); i++) {
			array[i] = scoreList.get(i).getId() + " (" + numberFormat.format(scoreList.get(i).getValue()) + ")";
		}
		return array;
	}
}

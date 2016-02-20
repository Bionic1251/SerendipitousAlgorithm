package recBrowser;

import java.util.HashMap;
import java.util.Map;

public class TestTrainReader extends BrowserFileReader {
	private final String userId;
	private Map<Long, Double> itemMap;

	public TestTrainReader(String userId) {
		itemMap = new HashMap<Long, Double>();
		this.userId = userId;
	}

	@Override
	protected void processLine(String line) {
		String[] brokenLine = line.split(",");
		String user = brokenLine[0];
		if (!user.equals(userId)) {
			return;
		}
		Long itemId = Long.valueOf(brokenLine[1]);
		Double rating = Double.valueOf(brokenLine[2]);
		itemMap.put(itemId, rating);
	}

	public Map<Long, Double> getItemMap() {
		return itemMap;
	}
}

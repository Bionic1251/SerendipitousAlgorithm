import java.io.BufferedReader;
import java.util.*;

public class CounterRunner {
	public static void main(String[] args) throws Exception {
		Map<Long, Integer> cumulativeMap = new HashMap<Long, Integer>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader("D:\\bigdata\\movielens\\ml-100k\\u.data"));
			//BufferedReader reader = new BufferedReader(new java.io.FileReader("D:\\bigdata\\movielens\\hetrec\\user_ratedmovies-timestamps.dat"));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] numbers = line.split("\t");
					String item = numbers[1];
					Long itemId = Long.valueOf(item);
					Integer num = 0;
					if (cumulativeMap.containsKey(itemId)) {
						num = cumulativeMap.get(itemId);
					}
					num++;
					cumulativeMap.put(itemId, num);
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		List<Integer> frequencyList = new ArrayList<Integer>(cumulativeMap.values());
		Collections.sort(frequencyList);
		Collections.reverse(frequencyList);
		for(Integer freq : frequencyList){
			System.out.println(freq);
		}
	}
}

package util;

import evaluationMetric.Container;
import org.grouplens.lenskit.util.statistics.MeanAccumulator;
import org.grouplens.lenskit.vectors.SparseVector;

import java.io.*;
import java.util.*;

public class PrepareUtil {

	public static void generateUnpopDataset(String ratingPath, int shortHead) {
		Map<String, Integer> popMap = getPopMap(ratingPath);
		Set<String> shortSet = getShortHeadItemIds(popMap, shortHead);
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(ratingPath));
			PrintWriter datasetWriter = new PrintWriter(new File("ratings.dat"));
			try {
				int i = 0;
				String line = reader.readLine();
				while (line != null) {
					i++;
					if (i % 100 == 0) {
						System.out.println(i + " processed");
					}
					String itemId = getItemId(line);
					if (!shortSet.contains(itemId)) {
						datasetWriter.println(line);
					}
					line = reader.readLine();
				}
			} finally {
				reader.close();
				datasetWriter.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static Set<String> getShortHeadItemIds(Map<String, Integer> popMap, int shortHead) {
		Set<String> shortSet = new HashSet<String>();
		List<Integer> freqList = new ArrayList<Integer>(popMap.values());
		Collections.sort(freqList);
		Collections.reverse(freqList);
		Integer smallestFreq = freqList.get(shortHead);
		for (Map.Entry<String, Integer> entry : popMap.entrySet()) {
			if (entry.getValue() >= smallestFreq) {
				shortSet.add(entry.getKey());
				if (shortSet.size() > shortHead) {
					break;
				}
			}
		}
		return shortSet;
	}

	public static void printUnpopularityRating(String ratingPath) {
		Map<String, Integer> popMap = getPopMap(ratingPath);
		int max = getMax(popMap);
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(ratingPath));
			PrintWriter writer = new PrintWriter(new File("unpop_rating.csv"));
			try {
				String line = reader.readLine();
				while (line != null) {
					String itemId = getItemId(line);
					int pop = popMap.get(itemId);
					double unpop = 1.0 - (double) pop / (double) max;
					String rating = getRating(line);
					writer.println(unpop + "," + rating);
					line = reader.readLine();
				}
			} finally {
				reader.close();
				writer.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static int getMax(Map<String, Integer> popMap) {
		int max = Integer.MIN_VALUE;
		for (Integer val : popMap.values()) {
			max = Math.max(max, val);
		}
		return max;
	}

	public static Map<String, Integer> getPopMap(String ratingPath) {
		Map<String, Integer> popMap = new HashMap<String, Integer>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(ratingPath));
			try {
				String line = reader.readLine();
				while (line != null) {
					String itemId = getItemId(line);
					int num = 0;
					if (popMap.containsKey(itemId)) {
						num = popMap.get(itemId);
					}
					num++;
					popMap.put(itemId, num);
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return popMap;
	}

	public static void printDissimilarityRating(String ratingPath, String contentPath) {
		int count = 0;
		ContentAverageDissimilarity.create(contentPath);
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(ratingPath));
			PrintWriter writer = new PrintWriter(new File("dissim_rating.csv"));
			try {
				Map<Long, Double> itemRatingMap = new HashMap<Long, Double>();
				String userId = "";
				String line = reader.readLine();
				while (line != null) {
					count++;
					if (count % 1000 == 0) {
						System.out.println("Processed " + count);
					}
					String newUserId = getUserId(line);
					if (!newUserId.equals(userId)) {
						printValues(itemRatingMap, writer);
						itemRatingMap = new HashMap<Long, Double>();
						userId = newUserId;
					}
					addRating(line, itemRatingMap);
					line = reader.readLine();
				}
			} finally {
				reader.close();
				writer.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void printValues(Map<Long, Double> itemRatingMap, PrintWriter writer) {
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> itemContentMap = dissimilarity.getItemContentMap();
		for (Map.Entry<Long, Double> item : itemRatingMap.entrySet()) {
			SparseVector contentItem = itemContentMap.get(item.getKey());
			double sum = 0;
			for (Map.Entry<Long, Double> tempItem : itemRatingMap.entrySet()) {
				if (item.getKey().equals(tempItem.getKey())) {
					continue;
				}
				SparseVector tempContentItem = itemContentMap.get(tempItem.getKey());
				double sim = ContentUtil.getSim(tempContentItem, contentItem);
				double dissim = 1 - sim;
				sum += dissim;
			}
			double res = sum / (double) (itemRatingMap.size() - 1);
			writer.println(res + "," + item.getValue());
		}
	}

	private static String getRating(String line) {
		String[] brokenLine = line.split("\t");
		return brokenLine[2];
	}

	private static String getItemId(String line) {
		String[] brokenLine = line.split("\t");
		return brokenLine[1];
	}

	private static String getUserId(String line) {
		String[] brokenLine = line.split("\t");
		return brokenLine[0];
	}

	private static void addRating(String line, Map<Long, Double> itemRating) {
		String[] brokenLine = line.split("\t");
		Long itemId = Long.valueOf(brokenLine[1]);
		Double rating = Double.valueOf(brokenLine[2]);
		itemRating.put(itemId, rating);
	}

	public static void printUserItemRatingNumber(String path) {
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				int i = 0;
				String line = reader.readLine();
				Set<Long> userSet = new HashSet<Long>();
				Set<Long> itemSet = new HashSet<Long>();
				while (line != null) {
					i++;
					String[] vec = line.split("\t");
					Long userId = Long.valueOf(vec[0]);
					userSet.add(userId);
					Long itemId = Long.valueOf(vec[1]);
					itemSet.add(itemId);
					line = reader.readLine();
				}
				System.out.println("Users " + userSet.size() + " Items " + itemSet.size() + " Ratings " + i);
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void prepareYahooDataset(String path) {
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			PrintWriter writer = new PrintWriter(new File("ratings.dat"));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] vector = line.split("\t");
					String row = vector[0] + "\t" + vector[1] + "\t" + vector[3];
					writer.println(row);
					line = reader.readLine();
				}
			} finally {
				reader.close();
				writer.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void printMoviesAndGetNames(String path, Map<String, String> map, String sep) {
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String text;
				String line = reader.readLine();
				while (line != null) {
					String[] vector = line.split(sep);
					text = vector[0];
					if (map.containsKey(text)) {
						map.put(text, vector[1]);
						System.out.println(line);
					}
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void getMoviesByName(String path, Map<String, String> map) {
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String text;
				String line = reader.readLine();
				while (line != null) {
					String[] vector = line.split("::");
					text = vector[1];
					String key = getKey(map, text);
					if (key != null) {
						System.out.println(line);
						map.put(key, line);
					}
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static String getKey(Map<String, String> map, String val) {
		for (Map.Entry<String, String> entry : map.entrySet()) {
			if (entry.getValue().equals(val)) {
				return entry.getKey();
			}
		}
		return null;
	}

	public static void prepareSmallDataset(String path) {
		int boolLen = 23, boolStart = 5;
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			PrintWriter writer = new PrintWriter(new File("small_content.dat"));
			try {
				String text;
				String line = reader.readLine();
				while (line != null) {
					String[] vector = line.split("\\|");
					text = vector[0];
					for (int i = boolStart; i <= boolLen; i++) {
						text += "," + vector[i];
					}
					writer.println(text);
					line = reader.readLine();
				}
			} finally {
				reader.close();
				writer.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void prepareBigDataset(String path) {
		Map<String, Integer> attributeMap = getAttributesMap(path);
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			PrintWriter writer = new PrintWriter(new File("big_content.dat"));
			try {
				String line = reader.readLine();
				String[] vector = line.split("\t");
				String id = vector[0];
				printTitle(attributeMap, writer);
				int[] values = new int[attributeMap.size()];
				while (line != null) {
					vector = line.split("\t");
					String newId = vector[0];
					if (!newId.equals(id)) {
						saveItem(id, values, writer);
						id = newId;
						values = new int[attributeMap.size()];
						continue;
					} else {
						if (!attributeMap.containsKey(vector[1])) {
							line = reader.readLine();
							continue;
						}
						int index = attributeMap.get(vector[1]);
						values[index] = 1;
					}

					line = reader.readLine();
					if (line == null) {
						saveItem(id, values, writer);
					}
				}
			} finally {
				reader.close();
				writer.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void printTitle(Map<String, Integer> attributeMap, PrintWriter writer) {
		String out = "id,";
		for (int i = 0; i < attributeMap.size(); i++) {
			for (Map.Entry<String, Integer> entry : attributeMap.entrySet()) {
				if (entry.getValue().equals(i)) {
					out += entry.getKey() + ",";
				}
			}
		}
		writer.println(out.substring(0, out.length() - 1));
	}

	private static void saveItem(String id, int[] values, PrintWriter writer) {
		String text = id;
		for (int i = 0; i < values.length; i++) {
			text += "," + values[i];
		}
		writer.println(text);
	}

	private static Map<String, Integer> getYahooAttributesMap(String path) {
		Set<String> set = new HashSet<String>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] vector = line.split("\t");
					String genres = vector[10];
					String[] genresVec = genres.split("\\|");
					for (String g : genresVec) {
						set.add(g);
					}
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		set.remove("\\N");
		set.remove("~Delete");
		List<String> list = new ArrayList<String>(set);
		Map<String, Integer> map = new HashMap<String, Integer>();
		for (int i = 0; i < list.size(); i++) {
			map.put(list.get(i), i);
		}
		return map;
	}

	private static Map<String, Integer> getAttributesMap(String path) {
		Set<String> set = new HashSet<String>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] vector = line.split("\t");
					set.add(vector[1]);
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		set.remove("IMAX");
		set.remove("Short");
		List<String> list = new ArrayList<String>(set);
		Map<String, Integer> map = new HashMap<String, Integer>();
		for (int i = 0; i < list.size(); i++) {
			map.put(list.get(i), i);
		}
		return map;
	}

	public static Map<Long, Double> getAverageRatingMap(String path) {
		Map<Long, Double> averageRatingMap = new HashMap<Long, Double>();
		Map<Long, MeanAccumulator> map = new HashMap<Long, MeanAccumulator>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] vector = line.split(",");
					long userId = Long.valueOf(vector[0]);
					double rating = Double.valueOf(vector[2]);
					MeanAccumulator meanAccumulator = new MeanAccumulator();
					if (map.containsKey(userId)) {
						meanAccumulator = map.get(userId);
					}
					meanAccumulator.add(rating);
					map.put(userId, meanAccumulator);
					line = reader.readLine();
				}
				for (Map.Entry<Long, MeanAccumulator> entry : map.entrySet()) {
					averageRatingMap.put(entry.getKey(), entry.getValue().getMean());
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return averageRatingMap;
	}

	public static void printItemPopularity(String path) {
		Map<String, Integer> popMap = PrepareUtil.getPopMap(path);
		Map<Integer, Integer> itemMap = new HashMap<Integer, Integer>();
		for (Integer val : popMap.values()) {
			int count = 0;
			if (itemMap.containsKey(val)) {
				count = itemMap.get(val);
			}
			count++;
			itemMap.put(val, count);
		}
		List<Integer> list = new ArrayList<Integer>(itemMap.keySet());
		Collections.sort(list);
		for (Integer freq : list) {
			System.out.println(freq + "\t" + itemMap.get(freq));
		}
	}

/*	public static void serrialize(MapStorage storage, String filename) {
		try {
			FileOutputStream fos = new FileOutputStream(filename);
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			oos.writeObject(storage);
			oos.flush();
			oos.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}*/

/*	public static MapStorage unserrialize(String filename) {
		try {
			FileInputStream fis = new FileInputStream(filename);
			ObjectInputStream oin = new ObjectInputStream(fis);
			MapStorage ts = (MapStorage) oin.readObject();
			return ts;
		} catch (Exception e) {
			System.out.println("Did not find " + filename);
		}
		return null;
	}*/

	public static void preprocess10M(String ratingPath, String contentPath) {
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(ratingPath));
			PrintWriter datasetWriter = new PrintWriter(new File("newRatings.dat"));
			try {
				Set<String> ids = getContentIds(contentPath);
				String line = reader.readLine();
				while (line != null) {
					String[] split = line.split("::");
					if (ids.contains(split[1])) {
						datasetWriter.println(split[0] + "\t" + split[1] + "\t" + split[2] + "\t" + split[3]);
					}
					line = reader.readLine();
				}
			} finally {
				reader.close();
				datasetWriter.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static Set<String> getContentIds(String contentPath) {
		Set<String> ids = new HashSet<String>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(contentPath));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] split = line.split(",");
					ids.add(split[0]);
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return ids;
	}

	public static Set<Long> getUserProfile(String path, long currentUserId) {
		Set<Long> itemSet = new HashSet<Long>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] vector = line.split(",");
					long userId = Long.valueOf(vector[0]);
					long itemId = Long.valueOf(vector[1]);
					if (userId == currentUserId) {
						itemSet.add(itemId);
					}
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return itemSet;
	}

	public static Map<Long, Double> getNormalizedPopMap(String ratingPath, String separator) {
		Map<Long, Double> popMap = new HashMap<Long, Double>();
		double max = 0;
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(ratingPath));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] split = line.split(separator);
					Long itemId = Long.valueOf(split[1]);
					double num = 0;
					if (popMap.containsKey(itemId)) {
						num = popMap.get(itemId);
					}
					num = num + 1;
					popMap.put(itemId, num);
					max = Math.max(max, num);
					line = reader.readLine();
				}
				Iterator<Map.Entry<Long, Double>> iterator = popMap.entrySet().iterator();
				while (iterator.hasNext()) {
					Map.Entry<Long, Double> entry = iterator.next();
					entry.setValue(entry.getValue() / max);
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return popMap;
	}
}

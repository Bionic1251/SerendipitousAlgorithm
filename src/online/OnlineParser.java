package online;

import evaluationMetric.Container;
import org.grouplens.lenskit.vectors.SparseVector;
import util.ContentAverageDissimilarity;
import util.ContentUtil;
import util.HTMLGenerator;
import util.PrepareUtil;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class OnlineParser {
	private static final int YEAR = 1980;
	private static final int LIST_SIZE = 10;
	private static final String[] PROHIBITED_ARRAY = {"162", "4307"};
	private static final String MOVIE_DATASET = "D:\\bigdata\\movielens\\hetrec\\movies.dat";

	public static void processInputData(String ratingPath) {
		Map<String, String> userMap = getUserList(ratingPath + "\\Users.txt");
		String out = "";
		for (Map.Entry<String, String> entry : userMap.entrySet()) {
			Set<Rating> ratingSet = getUserData(ratingPath + "\\" + entry.getValue() + ".csv", entry.getKey());
			validate(MOVIE_DATASET, ratingSet);
			for (Rating rating : ratingSet) {
				out += rating.getUserId() + "\t" + rating.getMovie().getId() + "\t" + rating.getRating() + "\t0" + "\n\r";
			}
		}
		System.out.println();
		System.out.println();
		System.out.println(out);
	}

	public static void processUserInputData(String path, String userId) {
		Set<Rating> ratingSet = getUserData(path, userId);
		validate(MOVIE_DATASET, ratingSet);
		generateUserData(ratingSet);
	}

	public static void generateFinalResults(String ansPath, String recPath, String ratingPath, String userId) {
		Map<String, RatedMovie> ratedMovies = getUserMovieList(ansPath);
		/*Map<String, RatedMovie> ratedMovies = new HashMap<String, RatedMovie>();
		Map<String, Integer> popMap = PrepareUtil.getPopMap(ratingPath);
		for(String key : popMap.keySet()){
			ratedMovies.put(key.toString(), new RatedMovie(key.toString(), "Name", true,true,true));
		}*/
		Set<String> itemIds = getUserItemIds(userId, ratingPath);
		Set<String> likedItemIds = getUserLikedItemIds(userId, ratingPath);
		fillRatedMovies(ratedMovies, itemIds, likedItemIds, "dataset/ml/big/content.dat", ratingPath);
		printFinalRecommendations(recPath, userId, ratedMovies);
	}

	private static void fillRatedMovies(Map<String, RatedMovie> ratedMovies, Set<String> profile, Set<String> likedItemIds, String contentPath, String ratingPath) {
		ContentAverageDissimilarity.create(contentPath);
		Map<String, Integer> popMap = PrepareUtil.getPopMap(ratingPath);
		double max = getMax(popMap);
		for (RatedMovie movie : ratedMovies.values()) {
			double unpop = 1 - (double) popMap.get(movie.getId()) / max;
			movie.setUnpopularity(unpop);
			double dissimilarity = getDissimilarity(profile, movie.getId());
			movie.setDissimilarity(dissimilarity);
			double likedDissimilarity = getDissimilarity(likedItemIds, movie.getId());
			movie.setLikedDissimilarity(likedDissimilarity);
		}
	}

	private static double getLikedDissimilarity(Set<String> profile, String itemId) {
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> map = dissimilarity.getItemContentMap();
		Long id = Long.valueOf(itemId);
		SparseVector vec = map.get(id);
		double dissim = 0;
		for (String ratedItem : profile) {
			Long ratedId = Long.valueOf(ratedItem);
			SparseVector ratedVec = map.get(ratedId);
			dissim += 1 - ContentUtil.getJaccard(ratedVec, vec);
		}
		return dissim / profile.size();
	}

	private static double getDissimilarity(Set<String> profile, String itemId) {
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> map = dissimilarity.getItemContentMap();
		Long id = Long.valueOf(itemId);
		SparseVector vec = map.get(id);
		double dissim = 0;
		for (String ratedItem : profile) {
			Long ratedId = Long.valueOf(ratedItem);
			SparseVector ratedVec = map.get(ratedId);
			dissim += 1 - ContentUtil.getJaccard(ratedVec, vec);
		}
		return dissim / profile.size();
	}

	private static double getMax(Map<String, Integer> popMap) {
		double max = 0;
		for (Integer val : popMap.values()) {
			max = Math.max(val, max);
		}
		return max;
	}

	private static Set<String> getUserItemIds(String userId, String ratingPath) {
		Set<String> itemIds = new HashSet<String>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(ratingPath));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] split = line.split("\t");
					String readUserId = split[0];
					if (readUserId.equals(userId)) {
						itemIds.add(split[1]);
					}
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return itemIds;
	}

	private static Set<String> getUserLikedItemIds(String userId, String ratingPath) {
		Set<String> itemIds = new HashSet<String>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(ratingPath));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] split = line.split("\t");
					String readUserId = split[0];
					Double rating = Double.valueOf(split[2]);
					if (readUserId.equals(userId) && rating >= 3.0) {
						itemIds.add(split[1]);
					}
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return itemIds;
	}

	private static void printFinalRecommendations(String recPath, String userId, Map<String, RatedMovie> ratedMovies) {
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(recPath));
			PrintWriter writer = new PrintWriter("finalResults.txt");
			try {
				String out = "";
				out += "relevant\tfamiliar\texpected\tU\tD\tLD\tid\tname\r\n";
				for (RatedMovie ratedMovie : ratedMovies.values()) {
					out += ratedMovie.getRelevant() + "\t" + ratedMovie.getNovel() + "\t" + ratedMovie.getUnexpected()
							+ "\t" + ratedMovie.getUnpopularity() + "\t" + ratedMovie.getDissimilarity() + "\t" + ratedMovie.getLikedDissimilarity()
							+ "\t" + ratedMovie.getId() + "\t" + ratedMovie.getName();
					out += "\r\n";
				}
				String line = reader.readLine();
				while (line != null) {
					String[] split = line.split("\t");
					String readUserId = split[1];
					if (!readUserId.equals(userId)) {
						line = reader.readLine();
						continue;
					}
					String algName = split[0];
					out += algName;
					out += "\r\n";
					out += "relevant\tfamiliar\texpected\tU\tD\tLD\tid\tname\r\n";
					String[] recs = split[2].split(",");
					for (String rec : recs) {
						String[] recSplit = rec.split("=");
						String id = recSplit[0];
						RatedMovie ratedMovie = ratedMovies.get(id);
						out += ratedMovie.getRelevant() + "\t" + ratedMovie.getNovel() + "\t" + ratedMovie.getUnexpected()
								+ "\t" + ratedMovie.getUnpopularity() + "\t" + ratedMovie.getDissimilarity() + "\t" + ratedMovie.getLikedDissimilarity()
								+ "\t" + id + "\t" + ratedMovie.getName();
						out += "\r\n";
					}
					line = reader.readLine();
				}
				writer.println(out);
				System.out.println(out);
			} finally {
				reader.close();
				writer.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}


	private static Map<String, RatedMovie> getUserMovieList(String ansPath) {
		Map<String, RatedMovie> ratedMovies = new HashMap<String, RatedMovie>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(ansPath));
			try {
				String line = reader.readLine();
				while (line != null) {
					if (line.length() == 0) {
						line = reader.readLine();
						continue;
					}
					String[] split = line.split("\t");
					String movieId = split[3];
					String movieName = split[4];
					Boolean relevant = split[0].equals("1");
					Boolean novel = split[1].equals("1");
					Boolean unexpected = split[2].equals("1");
					RatedMovie ratedMovie = new RatedMovie(movieId, movieName, relevant, novel, unexpected);
					ratedMovies.put(movieId, ratedMovie);
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return ratedMovies;
	}

	public static void generateRecommendationList(String path, String outPath) {
		Map<String, Set<String>> userMap = getUserRecommendations(path);
		printRecommendations(MOVIE_DATASET, userMap, outPath);
	}

	public static void checkTags(String path) {
		Map<Integer, Integer> tagWeightMap = new HashMap<Integer, Integer>();
		Map<Integer, Integer> tagMap = new HashMap<Integer, Integer>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				line = reader.readLine();
				while (line != null) {
					String[] split = line.split("\t");
					int tag = Integer.valueOf(split[1]);
					int num = 0;
					if (tagMap.containsKey(tag)) {
						num = tagMap.get(tag);
					}
					num++;
					tagMap.put(tag, num);
					int val = Integer.valueOf(split[2]);
					num = 0;
					if (tagWeightMap.containsKey(val)) {
						num = tagWeightMap.get(val);
					}
					num++;
					tagWeightMap.put(val, num);
					line = reader.readLine();
				}
				System.out.println("TagNum\tWeight");
				List<Integer> list = new ArrayList<Integer>(tagWeightMap.keySet());
				Collections.sort(list);
				for (Integer item : list) {
					System.out.println(item + "\t" + tagWeightMap.get(item));
				}
				System.out.println("Freq\tTagNum");
				Map<Integer, Integer> freqMap = new HashMap<Integer, Integer>();
				for (Map.Entry<Integer, Integer> tag : tagMap.entrySet()) {
					int num = 0;
					if (freqMap.containsKey(tag.getValue())) {
						num = freqMap.get(tag.getValue());
					}
					num++;
					freqMap.put(tag.getValue(), num);
				}
				list = new ArrayList<Integer>(freqMap.keySet());
				Collections.sort(list);
				for (Integer item : list) {
					System.out.println(item + "\t" + freqMap.get(item));
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void printNumberOfMoviesReleasedAfter(String path, int year) {
		int count = 0, all = 0;
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				line = reader.readLine();
				int min = Integer.MAX_VALUE;
				int max = Integer.MIN_VALUE;
				while (line != null) {
					String[] split = line.split("\t");
					int val = Integer.valueOf(split[5]);
					min = Math.min(min, val);
					max = Math.max(max, val);
					if (val >= year) {
						count++;
					}else{
						System.out.print(split[0] + "l,");
					}
					all++;
					line = reader.readLine();
				}
				System.out.println(min + " - " + max);
				System.out.println(count + " of " + all);
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void printRecommendations(String path, Map<String, Set<String>> userMap, String outPath) {
		for (Map.Entry<String, Set<String>> entry : userMap.entrySet()) {
			System.out.println(entry.getKey());
			try {
				BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
				PrintWriter writer = new PrintWriter(outPath + "\\" + entry.getKey() + ".html");
				try {
					String line = reader.readLine();
					line = reader.readLine();
					String out = HTMLGenerator.header;
					int i = 0;
					while (line != null) {
						String[] split = line.split("\t");
						if (entry.getValue().contains(split[0])) {
							//out += "0\t0\t0\t" + id + "\t" + name + "\t" + year + "\thttp://www.imdb.com/title/tt" + imdbId + "\t" + pic + "\t" + pic2;
							out += HTMLGenerator.getLine(split, i);
							i++;
						}
						line = reader.readLine();
					}
					out += HTMLGenerator.getBottom(i, entry.getKey());
					System.out.println(out);
					writer.println(out);
				} finally {
					reader.close();
					writer.close();
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	private static Map<String, Set<String>> getUserRecommendations(String path) {
		Map<String, Set<String>> userMap = new HashMap<String, Set<String>>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] split = line.split("\t");
					String userId = split[1];
					String[] recs = split[2].split(",");
					Set<String> userSet = new HashSet<String>();
					if (userMap.containsKey(userId)) {
						userSet = userMap.get(userId);
					}
					for (String rec : recs) {
						String[] recSplit = rec.split("=");
						String id = recSplit[0];
						userSet.add(id);
					}
					userMap.put(userId, userSet);
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return userMap;
	}

	public static void cleanRecommendations(String path) {
		Set<String> prohibited = new HashSet<String>(Arrays.asList(PROHIBITED_ARRAY));
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			PrintWriter writer = new PrintWriter("cleanRecommendations.txt");
			try {
				String line = reader.readLine();
				while (line != null) {
					String out = "";
					String[] split = line.split("\t");
					String algName = split[0];
					String userId = split[1];
					out += algName + "\t" + userId + "\t";
					String[] recs = split[2].split(",");
					int i = 0;
					for (String rec : recs) {
						String[] recSplit = rec.split("=");
						String id = recSplit[0];
						String score = recSplit[1];
						if (isMovieInRange(MOVIE_DATASET, id) && i < LIST_SIZE && !prohibited.contains(id)) {
							out += id + "=" + score + ",";
							i++;
						}
						if (i >= LIST_SIZE) {
							break;
						}
					}
					out = out.substring(0, out.length() - 1);
					writer.println(out);
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

	private static boolean isMovieInRange(String path, String id) {
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				line = reader.readLine();
				while (line != null) {
					String[] split = line.split("\t");
					String movieId = split[0];
					if (movieId.equals(id)) {
						int val = Integer.valueOf(split[5]);
						return val >= YEAR;
					}
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return false;
	}

	private static void generateUserData(Set<Rating> ratingSet) {
		for (Rating rating : ratingSet) {
			System.out.println(rating.getUserId() + "\t" + rating.getMovie().getId() + "\t" + rating.getRating() + "\t" + "0");
		}
	}

	public static void removeFunc(String path) {
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] split = line.split(" ");
					List<Container<Double>> list = new ArrayList<Container<Double>>();
					for (int i = 0; i < split.length; i++) {
						Double val = Double.valueOf(split[i]);
						list.add(new Container<Double>(Long.valueOf(i), val));
					}
					Collections.sort(list);
					for (int i = 0; i < 20; i++) {
						System.out.print(list.get(i).getId() + " ");
					}
					System.out.println();
					line = reader.readLine();
				}

			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void validate(String path, Set<Rating> ratings) {
		Map<String, Rating> ratingMap = new HashMap<String, Rating>();
		for (Rating rating : ratings) {
			ratingMap.put(rating.getMovie().getImdbId(), rating);
		}
		Set<String> newMovies = new HashSet<String>(ratingMap.keySet());
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				line = reader.readLine();
				while (line != null) {
					String[] split = line.split("\t");
					String imdbID = split[2];
					if (ratingMap.containsKey(imdbID)) {
						Rating rating = ratingMap.get(imdbID);
						rating.getMovie().setId(split[0]);
						System.out.println(line);
						newMovies.remove(imdbID);
					}
					line = reader.readLine();
				}
				Iterator<Rating> iterator = ratings.iterator();
				while (iterator.hasNext()) {
					Rating rating = iterator.next();
					if (newMovies.contains(rating.getMovie().getImdbId())) {
						System.out.println(rating.getMovie());
						iterator.remove();
					}
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static Map<String, String> getUserList(String path) {
		Map<String, String> userMap = new HashMap<String, String>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] split = line.split("\t");
					userMap.put(split[3], split[0]);
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return userMap;
	}

	private static Set<Rating> getUserData(String path, String userId) {
		Set<Rating> idList = new HashSet<Rating>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				line = reader.readLine();
				while (line != null) {
					String[] split = split(line);
					String imdbId = split[5];
					String name = split[0];
					String genres = split[9];
					Movie movie = new Movie(imdbId, name);
					movie.setGenres(genres);
					Double rating = Double.valueOf(split[1]);
					idList.add(new Rating(userId, movie, rating));
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return idList;
	}

	private static String[] split(String line) {
		Pattern p = Pattern.compile("\".+?\"");
		Matcher m = p.matcher(line);
		while (m.find()) {
			String str = m.group();
			String clearStr = str.replace(",", "");
			line = line.replace(str, clearStr);
		}
		return line.split(",");
	}
}


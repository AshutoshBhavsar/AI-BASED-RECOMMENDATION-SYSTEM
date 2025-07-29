package com.example;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class RecommenderApp {

    // A map to hold all user ratings: UserID -> (ItemID -> Rating)
    private static Map<Integer, Map<Integer, Double>> allUserRatings = new HashMap<>();

    public static void main(String[] args) {
        try {
            // Load the data from the CSV file
            loadData("data/dataset.csv");

            // The user for whom we want to generate recommendations
            int targetUser = 1;

            // Get the recommendations
            List<Map.Entry<Integer, Double>> recommendations = getRecommendations(targetUser);

            // Print the top 5 recommendations
            System.out.println("Top 5 recommendations for user " + targetUser + ":");
            recommendations.stream()
                    .limit(5)
                    .forEach(rec -> System.out.printf("Item: %d, Predicted Score: %.4f\n", rec.getKey(), rec.getValue()));

        } catch (IOException e) {
            System.out.println("Error reading the data file.");
            e.printStackTrace();
        }
    }

    /**
     * Generates a list of recommended items for a target user.
     */
    private static List<Map.Entry<Integer, Double>> getRecommendations(int targetUser) {
        // Step 1: Find similarity scores between the target user and all other users
        Map<Integer, Double> similarityScores = new HashMap<>();
        for (int otherUser : allUserRatings.keySet()) {
            if (otherUser != targetUser) {
                double similarity = calculatePearsonSimilarity(targetUser, otherUser);
                if (!Double.isNaN(similarity) && similarity > 0) { // Only consider positive correlations
                    similarityScores.put(otherUser, similarity);
                }
            }
        }

        // Step 2: Get the top N most similar users (the "neighborhood")
        List<Map.Entry<Integer, Double>> neighbors = similarityScores.entrySet().stream()
                .sorted(Map.Entry.<Integer, Double>comparingByValue().reversed())
                .limit(10) // Let's consider the top 10 neighbors
                .collect(Collectors.toList());

        // Step 3: Calculate predicted scores for items the target user hasn't rated yet
        Map<Integer, Double> recommendedScores = new HashMap<>();
        Set<Integer> targetUserItems = allUserRatings.get(targetUser).keySet();

        for (Map.Entry<Integer, Double> neighborEntry : neighbors) {
            int neighborUser = neighborEntry.getKey();
            double neighborSimilarity = neighborEntry.getValue();
            Map<Integer, Double> neighborRatings = allUserRatings.get(neighborUser);

            for (Map.Entry<Integer, Double> ratingEntry : neighborRatings.entrySet()) {
                int item = ratingEntry.getKey();
                // If the target user has NOT rated this item
                if (!targetUserItems.contains(item)) {
                    // Calculate the weighted score
                    double weightedRating = neighborSimilarity * ratingEntry.getValue();
                    recommendedScores.merge(item, weightedRating, Double::sum);
                }
            }
        }

        // Step 4: Sort the recommendations by predicted score in descending order
        return recommendedScores.entrySet().stream()
                .sorted(Map.Entry.<Integer, Double>comparingByValue().reversed())
                .collect(Collectors.toList());
    }

    /**
     * Calculates the Pearson Correlation similarity between two users.
     */
    private static double calculatePearsonSimilarity(int user1, int user2) {
        Map<Integer, Double> ratings1 = allUserRatings.get(user1);
        Map<Integer, Double> ratings2 = allUserRatings.get(user2);

        // Find items that both users have rated
        Set<Integer> commonItems = new HashSet<>(ratings1.keySet());
        commonItems.retainAll(ratings2.keySet());

        if (commonItems.size() < 2) { // Need at least 2 common items for a meaningful correlation
            return Double.NaN;
        }

        double[] user1Vec = new double[commonItems.size()];
        double[] user2Vec = new double[commonItems.size()];
        int i = 0;
        for (int item : commonItems) {
            user1Vec[i] = ratings1.get(item);
            user2Vec[i] = ratings2.get(item);
            i++;
        }

        return new PearsonsCorrelation().correlation(user1Vec, user2Vec);
    }

    /**
     * Loads the rating data from a CSV file into our map.
     */
    private static void loadData(String filePath) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                int userId = Integer.parseInt(values[0]);
                int itemId = Integer.parseInt(values[1]);
                double rating = Double.parseDouble(values[2]);

                allUserRatings.computeIfAbsent(userId, k -> new HashMap<>()).put(itemId, rating);
            }
        }
    }
}

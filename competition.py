from pyspark import SparkContext
import sys, time, math, csv, json
import xgboost as xgb
from datetime import datetime


"""
Method Description:
I used XGBregressor (a regressor based on Decision Tree) to train the model and predict the stars.
The way I decrease the RMSE is to add more features to the model. I added almost all the numerical features from the business.json and user.json files.
I convert the date in the user.json file to the number of days since 1970-01-01 and I also added the number of hours the business is open in a week.
I used one-hot encoding for the categories and label encoding for the city and state.

Error Distribution:
>=0 and <1: 102006
>=1 and <2: 33115
>=2 and <3: 6117
>=3 and <4: 806
>=4: 0

RMSE:
0.9796879126858766

Execution Time:
103.27411 seconds

"""


if __name__ == "__main__":

    start_time = time.time()

    folder_path = sys.argv[1]
    test_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    business_file = folder_path + "/business.json"
    user_file = folder_path + "/user.json"
    # train_file_path = folder_path + "/review_train.json"
    yelp_train_file_path = folder_path + "/yelp_train.csv"


    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    data = sc.textFile(business_file).map(lambda line: json.loads(line))
    def preprocess_categories(categories):
        if categories is None:
            return []
        else:
            return [cat.strip() for cat in categories.split(',')]
        
    data = data.map(lambda x: (x['business_id'], preprocess_categories(x['categories']), x['city'], x['state']))

    # Label encoding
    all_categories = sorted(data.flatMap(lambda x: x[1]).distinct().collect())
    cat_label_mapping = {category: idx for idx, category in enumerate(all_categories)}
    
    all_city = sorted(data.map(lambda x: x[2]).distinct().collect())
    city_label_mapping = {city: idx for idx, city in enumerate(all_city)}

    all_state = sorted(data.map(lambda x: x[3]).distinct().collect())
    state_label_mapping = {state: idx for idx, state in enumerate(all_state)}

    def label_encode_categories(categories):
        return [cat_label_mapping[cat] for cat in categories]

    # One-hot encoding
    def one_hot_encode_categories(categories):
        return [1 if cat in categories else 0 for cat in all_categories]

    # label_encoded_data = data.map(lambda x: (x[0], label_encode_categories(x[1])))
    # business_label_encoded_data = label_encoded_data.collectAsMap()

    # one_hot_encoded_data = data.map(lambda x: (x[0], one_hot_encode_categories(x[1])))



    def find_hours(hours):
        num = 0
        if hours is None:
            return num
        # days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        days = ["Friday", "Saturday", "Sunday"]
        for day in days:
            if day in hours:
                num += 1
        return num



    business_rdd = sc.textFile(business_file).map(lambda x: json.loads(x)).map(lambda x: (x["business_id"], (x["review_count"], float(x["stars"]), sum(label_encode_categories(preprocess_categories(x["categories"]))), city_label_mapping[x["city"]], state_label_mapping[x["state"]], x["is_open"], find_hours(x["hours"]) )))
    
    user_rdd = sc.textFile(user_file).map(lambda x: json.loads(x)).map(lambda x: (x["user_id"], (x["review_count"], float(x["average_stars"]), x["yelping_since"], float(x["useful"]), float(x['funny']), x['cool'], x['compliment_hot'], x['compliment_more'], x['compliment_profile'], x['compliment_cute'], x['compliment_list'], x['compliment_note'], x['compliment_plain'], x['compliment_cool'], x['compliment_funny'], x['compliment_writer'], x['compliment_photos'] )))
    
    business_dict = business_rdd.collectAsMap()
    user_dict = user_rdd.collectAsMap()

    # print("business_rdd: ", business_rdd.take(5))


    train_rdd = sc.textFile(yelp_train_file_path).filter(lambda x: "user_id" not in x).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1], float(x[2])))
    # test_rdd = sc.textFile(test_file_path).filter(lambda x: "user_id" not in x).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1], float(x[2])))
    test_rdd = sc.textFile(test_file_path).filter(lambda x: "user_id" not in x).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1], 0))


    date_format = "%Y-%m-%d"


    # Define a function to extract features from the data:
    def extract_features(user, business, stars):
        user_info = user_dict.get(user, (0, 0, 0))
        user_review_count = user_info[0]
        user_avg_stars = user_info[1]

        user_yelping_since = user_info[2]
        date_object = datetime.strptime(user_yelping_since, date_format)
        user_yelping_since = (date_object - datetime(1970, 1, 1)).days

        user_useful = user_info[3]
        user_funny = user_info[4]
        user_cool = user_info[5]
        user_compliment_hot = user_info[6]
        user_compliment_more = user_info[7]
        user_compliment_profile = user_info[8]
        user_compliment_cute = user_info[9]
        user_compliment_list = user_info[10]
        user_compliment_note = user_info[11]
        user_compliment_plain = user_info[12]
        user_compliment_cool = user_info[13]
        user_compliment_funny = user_info[14]
        user_compliment_writer = user_info[15]
        user_compliment_photos = user_info[16]


        # business_info = business_dict.get(business, (0, 0, 0))
        business_info = business_dict.get(business, (0, 0, 0, 0, 0))
        business_review_count = business_info[0]
        business_avg_stars = business_info[1]

        business_categories = business_info[2]
        business_city = business_info[3]
        business_state = business_info[4]
        business_is_open = business_info[5]
        business_hours = business_info[6]


        features = [user_avg_stars, user_review_count, user_yelping_since, user_useful, user_funny, user_cool, user_compliment_hot, user_compliment_more, user_compliment_profile, user_compliment_cute, user_compliment_list, user_compliment_note, user_compliment_plain, user_compliment_cool, user_compliment_funny, user_compliment_writer, user_compliment_photos, business_avg_stars, business_review_count, business_categories, business_state, business_city, business_is_open, business_hours]
        
        return (features, stars)





    # Apply the function to create feature vectors and labels:
    # train_features: [ ( [user_avg_stars, user_review_count, business_avg_stars, business_review_count], stars ), ... ]
    train_features = train_rdd.map(lambda x: extract_features(x[0], x[1], x[2]))
    test_features = test_rdd.map(lambda x: extract_features(x[0], x[1], x[2]))


    # Prepare data for XGBRegressor:
    X_train = train_features.map(lambda x: x[0]).collect()
    y_train = train_features.map(lambda x: x[1]).collect()

    X_test = test_features.map(lambda x: x[0]).collect()
    y_test = test_features.map(lambda x: x[1]).collect()


    d_train = xgb.DMatrix(X_train, label=y_train)
    d_test = xgb.DMatrix(X_test, label=y_test)


    # Hyperparameters
    params = {
        'objective': 'reg:linear',
        'learning_rate': 0.14,
        'max_depth': 6,
        'min_child_weight': 3,
        'lambda': 3,
        'alpha': 5,
        'gamma': 0.8,
    }
    n_estimators = 205





    print("----------------- train the model ------------------")
    model = xgb.train(params, d_train, num_boost_round=n_estimators)
    print("time: {0:.5f}".format(time.time() - start_time))



    print("----------------- predict the model ------------------")
    y_pred = model.predict(d_test)
    y_pred_rdd = sc.parallelize(y_pred).zipWithIndex().map(lambda x: (x[1], x[0]))

    test_predictions_rdd = test_rdd.map(lambda x: (x[0], x[1])).zipWithIndex().map(lambda x: (x[1], x[0])).join(y_pred_rdd).sortByKey().map(lambda x: (x[1][0][0], x[1][0][1], x[1][1]))
    print("time: {0:.5f}".format(time.time() - start_time))
    


    print("----------------- Collect Output List ------------------")
    test_predictions_list = test_predictions_rdd.collect()
    print("time: {0:.5f}".format(time.time() - start_time))



    print("----------------- wrtie output file ------------------")
    with open(output_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['user_id', 'business_id', 'prediction']
        csv_writer.writerow(header)
        csv_writer.writerows(test_predictions_list)

    print("time: {0:.5f}".format(time.time() - start_time))



    # print("----------------- Result ------------------")
    # # Calculate Mean Squared Error
    # def root_mean_squared_error(y_true, y_pred):
    #     error_dict = {}
    #     total = 0
    #     for i in range(len(y_true)):
    #         error = y_true[i] - y_pred[i]
    #         if 0 <= abs(error) < 1:
    #             error_dict[">=0 and <1"] = error_dict.get(">=0 and <1", 0) + 1
    #         elif 1 <= abs(error) < 2:
    #             error_dict[">=1 and <2"] = error_dict.get(">=1 and <2", 0) + 1
    #         elif 2 <= abs(error) < 3:
    #             error_dict[">=2 and <3"] = error_dict.get(">=2 and <3", 0) + 1
    #         elif 3 <= abs(error) < 4:
    #             error_dict[">=3 and <4"] = error_dict.get(">=3 and <4", 0) + 1
    #         elif 4 <= abs(error):
    #             error_dict[">=4"] = error_dict.get(">=4", 0) + 1
    #         total += error**2
    #     rmse = (total / len(y_true)) ** 0.5
    #     return rmse, error_dict
    
    # rmse, error_dict = root_mean_squared_error(y_test, y_pred)


    # # print("Parameters:", params, "num_boost_round:", n_estimators)
    # print("Error Distribution:", error_dict)
    # print("Root Mean Squared Error:", rmse)




# spark-submit competition.py "/Users/leoli/Desktop/HW3StudentData/" "/Users/leoli/Desktop/HW3StudentData/yelp_val.csv" "/Users/leoli/Desktop/HW3StudentData/competition_output.csv"





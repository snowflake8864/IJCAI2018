
# set the path-to-files
TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/valid.csv"
ONLINE_TRAIN_FILE = "data/online_train.csv"
ONLINE_TEST_FILE = "data/online_test.csv"
# TRAIN_FILE = "../data/train_v.csv"
# TEST_FILE = "../data/test_v.csv"
# DATA_FILE = "/home/icrcdpg/ICRC-chris/code/dnn_ctr-master/data/data.csv"

SUB_DIR = "./output"


NUM_SPLITS = 5
RANDOM_SEED = 19981015

# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
    # 'ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat',
    # 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat',
    # 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',
    # 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
    # 'ps_car_10_cat', 'ps_car_11_cat',
]

NUMERIC_COLS = [
    # numeric
    #'shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description',
    # feature engineering
    # 'item_id_PH_ctr','user_id_nexttime_delta','item_id_nexttime_delta',
    # 'item_id_lasttime_delta','shop_score_description','user_id_lasttime_delta',
    # 'item_property_C_count','shop_id_nexttime_delta','shop_id_lasttime_delta',
    # 'shop_id_PH_ctr','user_id_PH_ctr','user_item_id_lasttime_delta',
    # 'shop_id_count','item_property_list_3_count','tm_hour_sin'
]

IGNORE_COLS = ['is_trade','instance_id']
    #"instance_id", "is_trade", 
    #"context_id", "user_id"]

BASIC_COLS = ['is_trade','instance_id','item_collected_level', 'shop_id_count',
       'user_id_click_count_prep', 
       'user_item_id_nexttime_delta', 'user_brand_id_lasttime_delta',
       'item_price_level', 'user_category_id_count',
       'user_item_id_click_count_prep', 'user_occupation_id',
       'user_shop_id_lasttime_delta',
       'context_id',
       'user_id_nexttime_delta', 'item_sales_level',
       'item_id_nexttime_delta', 'item_id_lasttime_delta', 'tm_hour',
       'item_id', 'min', 'shop_score_description',
       'user_id_lasttime_delta','shop_id',
       'shop_id_nexttime_delta', 'item_brand_id', 'item_property_B_count',
       'item_id_click_count_prep', 'item_brand_id_nexttime_delta',
       'predict_category_property_B', 'mean_hour',
       'item_brand_id_lasttime_delta', 'shop_score_service',
       'shop_score_delivery', 'user_id_count', 'shop_id_click_count_prep',
       'user_category_id_nexttime_delta', 
       'item_id_count', 'shop_id_lasttime_delta',
       'user_category_id_lasttime_delta', 'item_property_A_count',
       'item_city_id', 'item_brand_id_click_count_prep',
       'shop_id_trade_prep_count', 'context_page_id',
       'user_item_id_lasttime_delta', 'item_id_trade_prep_count',
       'item_brand_id_trade_prep_count']
       # 'item_collected_level', 'shop_id_count',
       # 'user_id_click_count_prep']
STRONG_COLS = [
       'item_property_list_2_count',
       'item_property_list_1', 'shop_review_num_level',
       'item_brand_count', 'item_property_A_day_click',
       'item_property_list_1_count', 'shop_star_level',
       'item_property_B_1_day_click', 'user_query_day',
      'item_category2_day_click','shop_review_positive_rate',
       'predict_category_property_B_1', 'predict_category_property_A',
       'item_property_C_day_click', 'item_property_B_day_click',
       'tm_hour_cos', 'item_property_B_1_day_hour_click'
       'item_property_C_count',  'tm_hour_sin',
       'user_star_level', 'item_category2_count', 'item_property_list_3',
       'predict_category_property_C', 
       'user_age_level', 'item_city_count', 'item_property_list_3_count',
       'predict_category_property_A_1',
       'item_property_B_1_count', 'item_property_A_1_count',
       'item_property_list_2']

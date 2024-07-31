import tensorflow as tf
import pandas as pd
import numpy as np

def input_func(filename, num_edge_types, batch_size, neg_size=1, nbr_size=1,col_delim1=";", col_delim2=","):
    
    # # filename - file to fetch the data from
    
    # def_vals_s = [[-1], [-1]] + [['']] + [['']] * 2 * num_edge_types + [[-1]] * num_edge_types
    # def_vals_t = [[-1], [-1]] + [['']] + [['']] * 2 * num_edge_types + [[-1]] * num_edge_types
    # def_vals = [[-1]] + def_vals_s + def_vals_t

    # # def_vals.length - 1+ 2*(3*num_edge_vals+3)
    # # def_vals = [[-1],[-1],[-1],[''],.........]

    # min_after_dequeue = min_after_times * batch_size
    # capacity = capacity_times * batch_size
    
    # # creates a text queue from all the files given within the array
    # # shuffle - true : shuffles the files within the input
    # # num_epochs - reads all the files num_epochs times 
    # data_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs, shuffle=shuffle)
    # reader = tf.TextLineReader()

    # # reads 'batch_size' records from data_queue - returns keys,values
    # _, value = reader.read_up_to(data_queue, batch_size)
    # value = tf.train.shuffle_batch(
    #     [value],
    #     batch_size=batch_size,
    #     num_threads=24,
    #     capacity=capacity, # max no of elements in the queue
    #     enqueue_many=True, # indicates that values has a batch of samples
    #     min_after_dequeue=min_after_dequeue)

    # infos = tf.decode_csv(value, record_defaults=def_vals, field_delim=col_delim1)

    infos = pd.read_csv(filename, delimiter=col_delim1,header=None).to_numpy()

    e_types = tf.convert_to_tensor(np.array(infos[:,0],dtype=int), dtype=tf.int32)
    s_ids = tf.convert_to_tensor(np.array(infos[:,1],dtype=int), dtype=tf.int32)
    s_types = tf.convert_to_tensor(np.array(infos[:,2],dtype=int), dtype=tf.int32)
    s_negs = decode_neg_ids(infos[:,3], neg_size, batch_size, col_delim2)

    s_nbr_infos = decode_nbr_infos(batch_size, nbr_size, infos[:,4:4 + num_edge_types],
                                   infos[:,4 + num_edge_types:4 + 2 * num_edge_types],
                                   infos[:,4 + 2 * num_edge_types:4 + 3 * num_edge_types])

    t_ids = tf.convert_to_tensor(np.array(infos[:,4 + 3 * num_edge_types],dtype=int), dtype=tf.int32)
    t_types = tf.convert_to_tensor(np.array(infos[:,5 + 3 * num_edge_types],dtype=int), dtype=tf.int32)
    t_negs = decode_neg_ids(infos[:,6 + 3 * num_edge_types], neg_size, batch_size, col_delim2)
    base = 7 + 3 * num_edge_types
    t_nbr_infos = decode_nbr_infos(batch_size, nbr_size, infos[:,base:base + num_edge_types],
                                   infos[:,base + num_edge_types:base + 2 * num_edge_types],
                                   infos[:,base + 2 * num_edge_types:base + 3 * num_edge_types])

    train_data = [e_types, [s_ids, s_types, s_negs, s_nbr_infos], [t_ids, t_types, t_negs, t_nbr_infos]]

    return train_data


def decode_nbr_infos(batch_size, nbr_size, ids_list, weights_list, flags):
    type_nbr_info = []
    flags = tf.convert_to_tensor(flags, dtype = tf.int32)
    
    for e_type in range(ids_list.shape[1]):
        ids_str = ids_list[:,e_type]
        weight_str = weights_list[:,e_type]
        ids = decode_nbr_ids(ids_str, (batch_size, nbr_size))
        mask = decode_nbr_mask(ids_str, (batch_size, nbr_size))
        weights = decode_nbr_weights(weight_str, (batch_size, nbr_size))
        type_nbr_info.append([ids, mask, weights, flags[:,e_type]])
    return type_nbr_info


def decode_nbr_ids(str_np_arr, shape, delim=","):

    arr_nums = str_np_arr.ravel()
    # np.array([ np.array([]) if type(s) == float else np.array(s.split(',')).astype(int) for s in str_np_arr.ravel()])
    new_arr = np.zeros(shape, dtype=int)

    for i in range(len(arr_nums)):
      s = arr_nums[i]
      if not(type(s) == float):
        vals = np.array(s.split(',')).astype(float)
        for j in range(len(vals)):
          new_arr[i,j] = int(vals[j])

    # for i in range(arr_nums.shape[0]):
    #     for j in range(arr_nums[i].shape[0]): new_arr[i,j] = arr_nums[i][j]
    
    return tf.convert_to_tensor(new_arr)


def decode_nbr_mask(str_np_arr, shape, delim=","):
    
    arr_nums = str_np_arr.ravel()
    # np.array([ np.array([]) if type(s) == float else np.array(s.split(',')).astype(int) for s in str_np_arr.ravel()])
    new_arr = np.zeros(shape, dtype=int)

    for i in range(len(arr_nums)):
      s = arr_nums[i]
      if not(type(s) == float):
        vals = np.array(s.split(',')).astype(float)
        for j in range(len(vals)):
          new_arr[i,j] = 1
    
    return tf.convert_to_tensor(new_arr)


def decode_nbr_weights(str_np_arr, shape, delim=","):
    arr_nums = str_np_arr.ravel()
    # np.array([ np.array([]) if type(s) == float else np.array(s.split(',')).astype(int) for s in str_np_arr.ravel()])
    new_arr = np.zeros(shape, dtype=int)

    for i in range(len(arr_nums)):
      s = arr_nums[i]
      if not(type(s) == float):
        vals = np.array(s.split(',')).astype(float)
        for j in range(len(vals)):
          new_arr[i,j] = 1
    
    return tf.convert_to_tensor(new_arr)


def decode_neg_ids(str_np_arr, neg_size, batch_size, delim=","):
    
    arr_nums = str_np_arr.ravel()
    # np.array([ np.array([]) if type(s) == float else np.array(s.split(',')).astype(int) for s in str_np_arr.ravel()])
    new_arr = np.zeros((batch_size,neg_size), dtype=int)

    for i in range(len(arr_nums)):
      s = arr_nums[i]
      if not(type(s) == float):
        vals = np.array(s.split(',')).astype(float)
        for j in range(len(vals)):
          new_arr[i,j] = int(vals[j])
    
    return tf.convert_to_tensor(new_arr)
#include "thrust/device_vector.h"
#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS ipt 7:24:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:1:1
// %RANGE% CUB_DETAIL_L2_BACKOFF_NS l2b 0:1200:5
// %RANGE% CUB_DETAIL_L2_WRITE_LATENCY_NS l2w 0:1200:5

#if !TUNE_BASE
#if TUNE_TRANSPOSE == 0
#define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_DIRECT
#else // TUNE_TRANSPOSE == 1
#define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_WARP_TRANSPOSE
#endif // TUNE_TRANSPOSE

#if TUNE_LOAD == 0
#define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#else // TUNE_LOAD == 1
#define TUNE_LOAD_MODIFIER cub::LOAD_CA
#endif // TUNE_LOAD

struct device_reduce_by_key_policy_hub
{
  struct Policy350 : cub::ChainedPolicy<350, Policy350, Policy350>
  {
    using ReduceByKeyPolicyT = cub::AgentReduceByKeyPolicy<TUNE_THREADS,
                                                           TUNE_ITEMS,
                                                           TUNE_LOAD_ALGORITHM,
                                                           TUNE_LOAD_MODIFIER,
                                                           cub::BLOCK_SCAN_WARP_SCANS>;
  };

  using MaxPolicy = Policy350;
};
#endif // !TUNE_BASE

#include <cub/device/device_reduce.cuh>

template <class KeyT, class ValueT, class OffsetT>
static void reduce(nvbench::state &state, nvbench::type_list<KeyT, ValueT, OffsetT>)
{
  using keys_input_it_t = const KeyT*;
  using unique_output_it_t = KeyT*;
  using vals_input_it_t = const ValueT*;
  using aggregate_output_it_t = ValueT*;
  using num_runs_output_iterator_t = OffsetT*;
  using equality_op_t = cub::Equality;
  using reduction_op_t = cub::Sum;
  using accum_t = ValueT;
  using offset_t = OffsetT;

  #if !TUNE_BASE
  using dispatch_t = cub::DispatchReduceByKey<keys_input_it_t,
                                              unique_output_it_t,
                                              vals_input_it_t,
                                              aggregate_output_it_t,
                                              num_runs_output_iterator_t,
                                              equality_op_t,
                                              reduction_op_t,
                                              offset_t,
                                              accum_t,
                                              device_reduce_by_key_policy_hub>;
  #else
  using dispatch_t = cub::DispatchReduceByKey<keys_input_it_t,
                                              unique_output_it_t,
                                              vals_input_it_t,
                                              aggregate_output_it_t,
                                              num_runs_output_iterator_t,
                                              equality_op_t,
                                              reduction_op_t,
                                              offset_t,
                                              accum_t>;
  #endif

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const std::size_t min_segment_size = 1;
  const std::size_t max_segment_size = static_cast<std::size_t>(state.get_int64("MaxSegSize"));

  thrust::device_vector<OffsetT> num_runs_out(1);
  thrust::device_vector<ValueT> in_vals(elements);
  thrust::device_vector<ValueT> out_vals(elements);
  thrust::device_vector<KeyT> out_keys(elements);
  thrust::device_vector<KeyT> in_keys =
    gen_uniform_key_segments<KeyT>(seed_t{}, elements, min_segment_size, max_segment_size);

  KeyT *d_in_keys         = thrust::raw_pointer_cast(in_keys.data());
  KeyT *d_out_keys        = thrust::raw_pointer_cast(out_keys.data());
  ValueT *d_in_vals       = thrust::raw_pointer_cast(in_vals.data());
  ValueT *d_out_vals      = thrust::raw_pointer_cast(out_vals.data());
  OffsetT *d_num_runs_out = thrust::raw_pointer_cast(num_runs_out.data());

  std::uint8_t *d_temp_storage{};
  std::size_t temp_storage_bytes{};

  dispatch_t::Dispatch(d_temp_storage,
                       temp_storage_bytes,
                       d_in_keys,
                       d_out_keys,
                       d_in_vals,
                       d_out_vals,
                       d_num_runs_out,
                       equality_op_t{},
                       reduction_op_t{},
                       elements,
                       0);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  dispatch_t::Dispatch(d_temp_storage,
                       temp_storage_bytes,
                       d_in_keys,
                       d_out_keys,
                       d_in_vals,
                       d_out_vals,
                       d_num_runs_out,
                       equality_op_t{},
                       reduction_op_t{},
                       elements,
                       0);
  cudaDeviceSynchronize();
  const OffsetT num_runs = num_runs_out[0];

  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<ValueT>(num_runs);
  state.add_global_memory_writes<OffsetT>(1);

  std::cout << "runs: " << num_runs << std::endl;

  state.exec([&](nvbench::launch &launch) {
    dispatch_t::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_in_keys,
                         d_out_keys,
                         d_in_vals,
                         d_out_vals,
                         d_num_runs_out,
                         equality_op_t{},
                         reduction_op_t{},
                         elements,
                         launch.get_stream());
  });
}

using some_offset_types = nvbench::type_list<nvbench::int32_t>;

#ifdef TUNE_KeyT
using key_types = nvbench::type_list<TUNE_KeyT>;
#else // !defined(TUNE_KeyT)
using key_types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t, int128_t>;
#endif // TUNE_KeyT

#ifdef TUNE_ValueT
using value_types = nvbench::type_list<TUNE_ValueT>;
#else // !defined(TUNE_ValueT)
using value_types = all_types;
#endif // TUNE_ValueT

NVBENCH_BENCH_TYPES(reduce, NVBENCH_TYPE_AXES(key_types, value_types, some_offset_types))
  .set_name("cub::DeviceReduce::ReduceByKey")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegSize", {1, 4, 8});

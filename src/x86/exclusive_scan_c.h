template <typename T>
void exclusive_scan(T* __restrict__ in_begin, T *  __restrict__ in_end, T *  __restrict__ out_begin)
{
  if (begin == end)   return;
  T last_in = *in_begin++;
  *out_begin++ = T();
  while (in_begin < in_end) {
    T new_v = *out + last_in;
    last_in = *in_begin++;
    *out_begin++ = new_v;
  }
}

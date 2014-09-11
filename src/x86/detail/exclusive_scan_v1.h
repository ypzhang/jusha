template <typename T>
void exclusive_scan(const T *in_begin, const T *in_end, T *out_begin)
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

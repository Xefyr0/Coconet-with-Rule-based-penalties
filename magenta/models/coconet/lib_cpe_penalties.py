import tensorflow as tf
import numpy as np
import lib_graph

class CPEPenaltyCalculator:

  def __init__(self, hparams):
    self.hparams = hparams
    self.min_pitch = hparams.min_pitch
    self.max_pitch = hparams.max_pitch
    self.num_pitches = self.max_pitch-self.min_pitch+1
    self.num_instruments = hparams.num_instruments
    self.voice_range_matrix = self.construct_voice_range_matrix()
    self.voice_overlap_kernel = self.construct_voice_overlap_kernel()
    self.inharmonic_interval_kernel = self.construct_inharmonic_interval_kernel()
    self.perfect_octave_kernel = self.construct_perfect_octave_kernel()
    self.perfect_fifth_kernel = self.construct_perfect_fifth_kernel()


  def construct_voice_range_matrix(self, penalty_factor=1.):
    # We assume we're working on SATB format
    assert self.num_instruments==4
    max_instrument_range = self.max_pitch-self.min_pitch+1
    # Start out with all -1, then convert to zero inside valid range
    matrix = np.full([max_instrument_range, self.num_instruments], penalty_factor)
    for instrument in range(matrix.shape[0]):
      # All voices have a range of 19 semitones.
      for pitch in range(self.min_pitch, self.max_pitch+1):
        if instrument == 0 and pitch in range(60, 79+1):
          matrix[pitch-self.min_pitch, instrument] = 0
        if instrument == 1 and pitch in range(55, 74+1):
          matrix[pitch-self.min_pitch, instrument] = 0
        if instrument == 2 and pitch in range(46, 65+1):
          matrix[pitch-self.min_pitch, instrument] = 0
        if instrument == 3 and pitch in range(41, 60+1):
          matrix[pitch-self.min_pitch, instrument] = 0
    return tf.constant(matrix, dtype=tf.float32)

  @tf.function
  def calculate_voice_range_penalty(self, predictions):
    #tf.print("Voice range matrix: ", self.voice_range_matrix, summarize=-1)
    voice_range_penalties = tf.math.multiply(predictions, self.voice_range_matrix)
    #tf.print("Voice range multiplication results: ", voice_range_penalties[0, 0], summarize=-1)
    #tf.print("Voice range penalty scalar: ", tf.reduce_mean(voice_range_penalties))
    return voice_range_penalties

  def construct_voice_overlap_kernel(self, penalty_factor=0.1):
    # Construct matrix with convolution center
    matrix = np.zeros([1, 2*self.num_pitches-1, 2*self.num_instruments-1])
    center_note_index = self.num_pitches-1
    center_instrument_index = self.num_instruments-1

    for note in range(matrix.shape[-2]):
      for instrument in range(matrix.shape[-1]):
        # Higher instrument, lower note or lower instrument, higher note
        if (instrument < center_instrument_index and note < center_note_index) or (instrument > center_instrument_index and note > center_note_index):
          matrix[0, note, instrument] = abs(note - center_note_index)*penalty_factor
    return tf.constant(matrix, dtype=tf.float32)
  
  def construct_inharmonic_interval_kernel(self, penalty_factor=1.):
      # Construct matrix with convolution center
      matrix = np.zeros([1, 2*self.num_pitches-1, 2*self.num_instruments-1])
      center_note_index = self.num_pitches-1
      center_instrument_index = self.num_instruments-1

      for note in range(matrix.shape[-2]):
        for instrument in range(matrix.shape[-1]):
          # Penalty for inharmonic intervals
          if instrument != center_instrument_index and (abs(note-center_note_index)==1 or abs(note-center_note_index)==5):
            matrix[0, note, instrument] = penalty_factor
      return tf.constant(matrix, dtype=tf.float32)
  
  @tf.function
  def calculate_conv3d(self, predictions, kernel):
    reshaped_predictions=tf.expand_dims(predictions, axis=-1)
    #tf.print("Predictions shape: ", reshaped_predictions.shape)
    #tf.print("Kernel shape: ", kernel.shape)
    kernel_penalties=tf.nn.conv3d(
        input=reshaped_predictions,
        filters=kernel[..., tf.newaxis, tf.newaxis],
        strides=[1, 1, 1, 1, 1],
        padding="SAME")
    #tf.print("Voice overlap convolution result shape: ", kernal_penalties.shape)
    #tf.print("Voice overlap convolution result: ", kernel_penalties[0, 0], summarize=-1)
    return tf.math.multiply(predictions, tf.squeeze(kernel_penalties, axis=-1))
  
  @tf.function
  def calculate_kernel_penalty(self, predictions):
    #tf.print("Voice overlap kernel: ", self.voice_overlap_kernel, summarize=-1)
    #tf.print("Inharmonic interval kernel: ", self.inharmonic_interval_kernel, summarize=-1)
    kernel = self.voice_overlap_kernel+self.inharmonic_interval_kernel
    convresult = self.calculate_conv3d(predictions, kernel)
    #tf.print("kernel penalty result scalar: ", tf.reduce_mean(convresult))
    return convresult
    
  def construct_perfect_octave_kernel(self, penalty_factor=0.1):
      # Construct matrix with convolution center
      matrix = np.zeros([1, 2*self.num_pitches-1, 1])
      center_note_index = self.num_pitches-1

      for note in range(matrix.shape[0]):
        # Mark perfect octaves as nonzero
        if abs(note-center_note_index)%12==0:
          matrix[0, note, 0] = penalty_factor
      return tf.constant(matrix, dtype=tf.float32)
  
  def construct_perfect_fifth_kernel(self, penalty_factor=0.1):
      # Construct matrix with convolution center
      matrix = np.zeros([1, 2*self.num_pitches-1, 1])
      center_note_index = self.num_pitches-1

      for note in range(matrix.shape[-2]):
        # Mark perfect fifths as nonzero
        if abs(note-center_note_index)%12==7:
          matrix[0, note, 0] = penalty_factor
      return tf.constant(matrix, dtype=tf.float32)

  @tf.function
  def calculate_parallel_perfect_penalty(self, predictions):

    perfect_fifth_probability=self.calculate_conv3d(predictions, self.perfect_fifth_kernel)
    perfect_octave_probability=self.calculate_conv3d(predictions, self.perfect_octave_kernel)

    perfect_fifth_penalty=tf.zeros(shape=[self.hparams.batch_size, 1, self.num_pitches, self.num_pitches])
    perfect_octave_penalty=tf.zeros(shape=[self.hparams.batch_size, 1, self.num_pitches, self.num_pitches])

    for time in range(self.hparams.crop_piece_len-1):
      for ins_1_index in range(self.num_instruments):
        for ins_2_index in range(self.num_instruments):
          if ins_1_index != ins_2_index:
            perfect_fifth_penalty+=tf.matmul(
              perfect_fifth_probability[:, time, :, ins_1_index],
              perfect_fifth_probability[:, time+1, :, ins_2_index],
              transpose_b=True)
            perfect_octave_penalty+=tf.matmul(
              perfect_octave_probability[:, time, :, ins_1_index],
              perfect_octave_probability[:, time+1, :, ins_2_index],
              transpose_b=True)
    
    perfect_fifth_penalty-=tf.linalg.band_part(perfect_fifth_penalty, 0, 0)
    perfect_octave_penalty-=tf.linalg.band_part(perfect_octave_penalty, 0, 0)
    
    loss_scalar = tf.reduce_mean(perfect_fifth_probability+perfect_octave_probability)
    #tf.print("parallel/perfect octave loss scalar: ", loss_scalar)
    return loss_scalar

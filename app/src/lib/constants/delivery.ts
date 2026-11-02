/** Keep these instructions aligned with the controlled Ryan delivery samples. */
export const DELIVERY_PRESETS = [
  {
    id: 'neutral',
    instruct: 'Speak in a calm, neutral, matter-of-fact tone without strong emotion.',
  },
  {
    id: 'fear',
    instruct: 'Speak with clear fear and apprehension, a trembling voice and urgent breath.',
  },
  {
    id: 'happiness',
    instruct: 'Speak with clear happiness and delight, a warm smile and lively enthusiasm.',
  },
  {
    id: 'confusion',
    instruct:
      'Speak with clear confusion and uncertainty, hesitant pauses and questioning intonation.',
  },
] as const;

export function supportsDeliveryInstructions(
  engine: string | undefined,
  modelSize: string | undefined,
  modelSupportsInstruct?: boolean,
): boolean {
  return (
    engine === 'qwen_custom_voice' &&
    (modelSize ?? '1.7B') === '1.7B' &&
    modelSupportsInstruct !== false
  );
}

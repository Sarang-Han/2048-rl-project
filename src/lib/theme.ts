export const theme = {
  colors: {
    primary: {
      background: '#faf8ef',
      gradient: 'linear-gradient(135deg, #faf8ef 0%, #f7f4e9 100%)',
      text: '#776e65',
      textSecondary: '#8f7a66'
    },
    board: {
      background: 'linear-gradient(145deg, #c4b59f, #a89a82)',
      cellEmpty: 'rgba(206, 189, 166, 0.4)',
      shadow: '0 8px 24px rgba(0,0,0,0.12), inset 0 1px 0 rgba(255,255,255,0.2)'
    },
    controls: {
      background: 'linear-gradient(145deg, #ffffff, #f8f5f0)',
      shadow: '0 6px 20px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.8)',
      button: {
        primary: 'linear-gradient(145deg, #9f8a76, #8f7a66)',
        danger: 'linear-gradient(145deg, #f67c5f, #f65e3b)',
        secondary: 'linear-gradient(145deg, #d4c2ac, #c4b59f)'
      }
    },
    status: {
      success: '#22c55e',
      warning: '#f59e0b',
      error: '#ef4444',
      info: '#3b82f6'
    }
  },
  spacing: {
    boardSize: 440,
    padding: 24,
    gap: 12,
    borderRadius: {
      small: '12px',
      medium: '16px',
      large: '24px'
    }
  },
  animation: {
    fast: '150ms',
    normal: '300ms',
    slow: '500ms'
  }
} as const;